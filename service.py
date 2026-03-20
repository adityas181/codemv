from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import TYPE_CHECKING, Any

import aio_pika
from aio_pika.abc import AbstractIncomingMessage
from loguru import logger

from agentcore.services.base import Service
from agentcore.services.rabbitmq.config import RabbitMQConfig

if TYPE_CHECKING:
    from aio_pika import Channel, Connection, Queue
    from aio_pika.abc import AbstractRobustConnection


class RabbitMQService(Service):
    """RabbitMQ service for durable job scheduling with rate limiting.

    Option A implementation: consumers run inside the same FastAPI process.
    RabbitMQ provides durability, rate-limiting (prefetch_count),
    and visibility (management UI). The asyncio.Queue + EventManager + SSE
    streaming stays completely unchanged.

    Queues:
        - agentcore.build        : playground build jobs
        - agentcore.run          : run API + webhook jobs
        - agentcore.schedule     : cron/interval scheduled jobs
        - agentcore.trigger      : folder monitor + email monitor
        - agentcore.evaluation   : LLM judge evaluation jobs
        - agentcore.orchestrator : orchestrator streaming jobs
    """

    name = "rabbitmq_service"

    def __init__(self) -> None:
        self.config = RabbitMQConfig()
        self._connection: AbstractRobustConnection | None = None
        self._channel: Channel | None = None
        self._queues: dict[str, Queue] = {}
        self._consumer_tags: list[str] = []
        self._started = False
        self.ready = False

        # Stats tracking
        self._stats: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Connect to RabbitMQ, declare queues, and start consumers.

        Uses a single channel for all queues (CloudAMQP compatibility).
        Old messages from previous server runs are purged on startup since
        the in-memory job queues (EventManager, asyncio.Queue) are ephemeral
        and old job_ids cannot be processed.
        """
        if not self.config.enabled:
            logger.info("RabbitMQ is disabled (RABBITMQ_ENABLED != true). Skipping.")
            return

        try:
            # Mask password in URL for logging
            _safe_url = self.config.url
            try:
                from urllib.parse import urlparse
                _parsed = urlparse(self.config.url)
                if _parsed.password:
                    _safe_url = self.config.url.replace(_parsed.password, "****")
            except Exception:
                pass
            logger.info(f"Connecting to RabbitMQ: {_safe_url}")
            self._connection = await aio_pika.connect_robust(
                self.config.url,
                client_properties={"connection_name": "agentcore"},
            )
            logger.info("RabbitMQ connection established")

            self._channel = await self._connection.channel()
            await self._channel.set_qos(prefetch_count=self.config.prefetch_count)

            # All queues with their consumer handlers
            queue_consumers = [
                (self.config.build_queue, self._on_build_message),
                (self.config.run_queue, self._on_run_message),
                (self.config.schedule_queue, self._on_schedule_message),
                (self.config.trigger_queue, self._on_trigger_message),
                (self.config.orchestrator_queue, self._on_orchestrator_message),
            ]

            for queue_name, handler in queue_consumers:
                q = await self._channel.declare_queue(queue_name, durable=True)

                # Purge old messages — they reference job_ids from a previous
                # server session whose in-memory queues no longer exist.
                purge_result = await q.purge()
                msg_count = getattr(purge_result, "message_count", purge_result)
                if msg_count:
                    logger.info(f"Purged {msg_count} stale messages from {queue_name}")

                self._queues[queue_name] = q
                tag = await q.consume(handler)
                self._consumer_tags.append(tag)
                logger.info(f"Consumer registered on {queue_name} (tag={tag})")

                # Init stats for each queue
                short_name = queue_name.split(".")[-1]
                self._stats[f"{short_name}_published"] = 0
                self._stats[f"{short_name}_completed"] = 0
                self._stats[f"{short_name}_failed"] = 0

            self._started = True
            queue_names = ", ".join(self._queues.keys())
            logger.info(
                f"RabbitMQ started: queues=[{queue_names}], "
                f"prefetch={self.config.prefetch_count}"
            )
        except Exception:
            logger.exception("Failed to start RabbitMQ service")
            raise

    async def stop(self) -> None:
        """Gracefully close consumers and connection."""
        if not self._started:
            return

        for tag in self._consumer_tags:
            try:
                for q in self._queues.values():
                    await q.cancel(tag)
            except Exception:
                pass

        try:
            if self._channel and not self._channel.is_closed:
                await self._channel.close()
            if self._connection and not self._connection.is_closed:
                await self._connection.close()
        except Exception:
            logger.debug("Error closing RabbitMQ connection (may already be closed)")

        self._started = False
        logger.info(f"RabbitMQ service stopped. Stats: {self._stats}")

    async def teardown(self) -> None:
        await self.stop()

    def is_enabled(self) -> bool:
        return self.config.enabled and self._started

    def get_stats(self) -> dict[str, int]:
        return dict(self._stats)

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    async def publish_build_job(self, job_data: dict[str, Any]) -> str:
        return await self._publish(self.config.build_queue, job_data)

    async def publish_run_job(self, job_data: dict[str, Any]) -> str:
        return await self._publish(self.config.run_queue, job_data)

    async def publish_schedule_job(self, job_data: dict[str, Any]) -> str:
        return await self._publish(self.config.schedule_queue, job_data)

    async def publish_trigger_job(self, job_data: dict[str, Any]) -> str:
        return await self._publish(self.config.trigger_queue, job_data)

    async def publish_orchestrator_job(self, job_data: dict[str, Any]) -> str:
        return await self._publish(self.config.orchestrator_queue, job_data)

    async def _publish(self, queue_name: str, job_data: dict[str, Any]) -> str:
        if not self._channel or self._channel.is_closed:
            msg = "RabbitMQ channel is not available"
            raise RuntimeError(msg)

        message_id = job_data.get("job_id", str(uuid.uuid4()))
        body = json.dumps(job_data, default=str).encode("utf-8")

        message = aio_pika.Message(
            body=body,
            message_id=message_id,
            content_type="application/json",
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
        )

        await self._channel.default_exchange.publish(message, routing_key=queue_name)

        short_name = queue_name.split(".")[-1]
        self._stats[f"{short_name}_published"] = self._stats.get(f"{short_name}_published", 0) + 1
        logger.info(f"[RabbitMQ] Published job {message_id} to {queue_name}")
        return message_id

    # ------------------------------------------------------------------
    # Consumer helpers
    # ------------------------------------------------------------------

    def _track(self, queue_name: str, status: str) -> None:
        short_name = queue_name.split(".")[-1]
        key = f"{short_name}_{status}"
        self._stats[key] = self._stats.get(key, 0) + 1

    # ------------------------------------------------------------------
    # Consumers
    # ------------------------------------------------------------------

    async def _safe_process(
        self,
        message: AbstractIncomingMessage,
        queue_name: str,
        handler,
    ) -> None:
        """Process a message safely — no exception can escape and kill the consumer.

        The try/except wraps the ENTIRE message.process() context manager so that
        failures from message.ack() / message.reject() inside __aexit__ are also
        caught.  Any unhandled exception (including CancelledError) that leaks
        into aio_pika's consumer framework permanently kills the consumer.
        """
        job_id = None
        start_time = time.time()
        try:
            async with message.process():
                job_id = await handler(message, start_time)
            logger.debug(f"[RabbitMQ] Message acked for {queue_name} job {job_id}")
        except asyncio.CancelledError:
            self._track(queue_name, "failed")
            logger.warning(
                f"[RabbitMQ] {queue_name} job cancelled: {job_id} "
                f"({time.time() - start_time:.2f}s)"
            )
        except Exception:
            self._track(queue_name, "failed")
            logger.exception(
                f"[RabbitMQ] {queue_name} job failed (message processing error): "
                f"{job_id} ({time.time() - start_time:.2f}s)"
            )

    async def _on_build_message(self, message: AbstractIncomingMessage) -> None:
        logger.info(f"[RabbitMQ] >>> Build message RECEIVED (delivery_tag={message.delivery_tag})")
        await self._safe_process(message, self.config.build_queue, self._handle_build)

    async def _handle_build(self, message: AbstractIncomingMessage, start_time: float) -> str:
        from agentcore.services.deps import get_queue_service

        job_data = json.loads(message.body.decode("utf-8"))
        job_id = job_data["job_id"]
        logger.info(f"[RabbitMQ] Processing build job: {job_id}")

        queue_service = get_queue_service()
        _, event_manager, _, _ = queue_service.get_queue_data(job_id)

        await self._execute_build_job(job_data, event_manager, queue_service)

        _, _, task, _ = queue_service.get_queue_data(job_id)
        if task and not task.done():
            await task

        self._track(self.config.build_queue, "completed")
        logger.info(f"[RabbitMQ] Build job completed: {job_id} ({time.time() - start_time:.2f}s)")
        return job_id

    async def _on_run_message(self, message: AbstractIncomingMessage) -> None:
        await self._safe_process(message, self.config.run_queue, self._handle_run)

    async def _handle_run(self, message: AbstractIncomingMessage, start_time: float) -> str:
        from agentcore.services.deps import get_queue_service

        job_data = json.loads(message.body.decode("utf-8"))
        job_id = job_data["job_id"]
        logger.info(f"[RabbitMQ] Processing run job: {job_id}")

        queue_service = get_queue_service()
        _, event_manager, _, _ = queue_service.get_queue_data(job_id)

        await self._execute_run_job(job_data, event_manager, queue_service)

        self._track(self.config.run_queue, "completed")
        logger.info(f"[RabbitMQ] Run job completed: {job_id} ({time.time() - start_time:.2f}s)")
        return job_id

    async def _on_schedule_message(self, message: AbstractIncomingMessage) -> None:
        await self._safe_process(message, self.config.schedule_queue, self._handle_schedule)

    async def _handle_schedule(self, message: AbstractIncomingMessage, start_time: float) -> str:
        from agentcore.services.deps import get_scheduler_service

        job_data = json.loads(message.body.decode("utf-8"))
        job_id = f"agent={job_data['agent_id']}"
        logger.info(
            f"[RabbitMQ] Processing schedule job: agent={job_data['agent_id']} "
            f"trigger={job_data['trigger_config_id']}"
        )

        scheduler_service = get_scheduler_service()
        await scheduler_service._execute_trigger_direct(
            trigger_config_id=uuid.UUID(job_data["trigger_config_id"]),
            agent_id=uuid.UUID(job_data["agent_id"]),
            environment=job_data.get("environment", "dev"),
            version=job_data.get("version"),
        )

        self._track(self.config.schedule_queue, "completed")
        logger.info(f"[RabbitMQ] Schedule job completed: {job_id} ({time.time() - start_time:.2f}s)")
        return job_id

    async def _on_trigger_message(self, message: AbstractIncomingMessage) -> None:
        await self._safe_process(message, self.config.trigger_queue, self._handle_trigger)

    async def _handle_trigger(self, message: AbstractIncomingMessage, start_time: float) -> str:
        from agentcore.services.deps import get_trigger_service

        job_data = json.loads(message.body.decode("utf-8"))
        trigger_type = job_data.get("trigger_type", "unknown")
        job_id = f"agent={job_data['agent_id']}"
        logger.info(
            f"[RabbitMQ] Processing {trigger_type} trigger: agent={job_data['agent_id']} "
            f"trigger={job_data['trigger_config_id']}"
        )

        trigger_service = get_trigger_service()
        await trigger_service._execute_trigger_direct(
            trigger_config_id=uuid.UUID(job_data["trigger_config_id"]),
            agent_id=uuid.UUID(job_data["agent_id"]),
            payload=job_data.get("payload", {}),
            environment=job_data.get("environment", "dev"),
            version=job_data.get("version"),
            trigger_config=job_data.get("trigger_config"),
        )

        self._track(self.config.trigger_queue, "completed")
        logger.info(f"[RabbitMQ] {trigger_type} trigger completed: {job_id} ({time.time() - start_time:.2f}s)")
        return job_id

    async def _on_orchestrator_message(self, message: AbstractIncomingMessage) -> None:
        await self._safe_process(message, self.config.orchestrator_queue, self._handle_orchestrator)

    async def _handle_orchestrator(self, message: AbstractIncomingMessage, start_time: float) -> str:
        from agentcore.services.deps import get_queue_service

        job_data = json.loads(message.body.decode("utf-8"))
        job_id = job_data["job_id"]
        logger.info(f"[RabbitMQ] Processing orchestrator job: {job_id}")

        queue_service = get_queue_service()
        _, event_manager, _, _ = queue_service.get_queue_data(job_id)

        await self._execute_orchestrator_job(job_data, event_manager)

        self._track(self.config.orchestrator_queue, "completed")
        logger.info(f"[RabbitMQ] Orchestrator job completed: {job_id} ({time.time() - start_time:.2f}s)")
        return job_id

    # ------------------------------------------------------------------
    # Job executors
    # ------------------------------------------------------------------

    async def _execute_build_job(self, job_data: dict[str, Any], event_manager: Any, queue_service: Any) -> None:
        from fastapi import BackgroundTasks

        from agentcore.api.build import generate_agent_events
        from agentcore.api.v1_schemas import AgentDataRequest, InputValueRequest
        from agentcore.services.database.models.user.model import User
        from agentcore.services.deps import session_scope

        job_id = job_data["job_id"]
        agent_id = uuid.UUID(job_data["agent_id"])

        inputs = InputValueRequest(**job_data["inputs"]) if job_data.get("inputs") else None
        data = AgentDataRequest(**job_data["data"]) if job_data.get("data") else None

        user_id = job_data.get("user_id")
        async with session_scope() as session:
            current_user = await session.get(User, uuid.UUID(user_id)) if user_id else None

        if current_user is None:
            logger.error(f"[RabbitMQ] User not found for build job {job_id}")
            return

        background_tasks = BackgroundTasks()
        task_coro = generate_agent_events(
            agent_id=agent_id,
            background_tasks=background_tasks,
            event_manager=event_manager,
            inputs=inputs,
            data=data,
            files=job_data.get("files"),
            stop_component_id=job_data.get("stop_component_id"),
            start_component_id=job_data.get("start_component_id"),
            log_builds=job_data.get("log_builds", True),
            current_user=current_user,
            agent_name=job_data.get("agent_name"),
        )
        queue_service.start_job(job_id, task_coro)

        # Signal the placeholder task (if any) that the real job has started
        job_ready = getattr(event_manager, "_job_ready", None)
        if job_ready is not None:
            job_ready.set()

    async def _execute_run_job(self, job_data: dict[str, Any], event_manager: Any, queue_service: Any) -> None:
        """Execute a run job. Handles both streaming and non-streaming."""
        from agentcore.api.endpoints import simple_run_agent
        from agentcore.api.v1_schemas import SimplifiedAPIRequest
        from agentcore.services.database.models.agent.model import Agent
        from agentcore.services.deps import session_scope

        job_id = job_data["job_id"]
        agent_id = uuid.UUID(job_data["agent_id"])
        is_stream = job_data.get("stream", True)

        async with session_scope() as session:
            agent = await session.get(Agent, agent_id)

        if agent is None:
            logger.error(f"[RabbitMQ] Agent not found for run job {job_id}")
            return

        if job_data.get("agent_data"):
            agent.data = job_data["agent_data"]

        input_request = SimplifiedAPIRequest(**job_data.get("input_request", {}))

        prod_deployment = None
        uat_deployment = None
        if job_data.get("prod_deployment_id"):
            from agentcore.services.database.models.agent_deployment_prod.model import AgentDeploymentProd
            async with session_scope() as session:
                prod_deployment = await session.get(AgentDeploymentProd, uuid.UUID(job_data["prod_deployment_id"]))
        if job_data.get("uat_deployment_id"):
            from agentcore.services.database.models.agent_deployment_uat.model import AgentDeploymentUAT
            async with session_scope() as session:
                uat_deployment = await session.get(AgentDeploymentUAT, uuid.UUID(job_data["uat_deployment_id"]))

        if is_stream:
            # Streaming: run the agent directly with event_manager for token streaming.
            # Do NOT use run_agent_generator here — it waits on a client_consumed_queue
            # that only the HTTP streaming response writes to.  In the RabbitMQ path the
            # HTTP response reads from the shared asyncio.Queue independently, so the
            # consumer must not block on client consumption.
            try:
                result = await simple_run_agent(
                    agent=agent,
                    input_request=input_request,
                    stream=True,
                    api_key_user=None,
                    event_manager=event_manager,
                    prod_deployment=prod_deployment,
                    uat_deployment=uat_deployment,
                )
                event_manager.on_end(data={"result": result.model_dump()})
            except Exception as exc:
                logger.exception(f"[RabbitMQ] Streaming run job error: {job_id}")
                event_manager.on_error(data={"error": str(exc)})
            finally:
                await event_manager.queue.put((None, None, time.time()))
        else:
            # Non-streaming: run agent directly and send result back via queue
            try:
                result = await simple_run_agent(
                    agent=agent,
                    input_request=input_request,
                    stream=False,
                    api_key_user=None,
                    prod_deployment=prod_deployment,
                    uat_deployment=uat_deployment,
                )
                result_event = json.dumps({"event": "end", "data": {"result": result.model_dump()}}, default=str) + "\n\n"
                event_manager.queue.put_nowait(("end", result_event.encode("utf-8"), time.time()))
            except Exception as exc:
                error_event = json.dumps({"event": "error", "data": {"error": str(exc)}}) + "\n\n"
                event_manager.queue.put_nowait(("error", error_event.encode("utf-8"), time.time()))
            finally:
                event_manager.queue.put_nowait((None, None, time.time()))

    async def _execute_orchestrator_job(self, job_data: dict[str, Any], event_manager: Any) -> None:
        from agentcore.api.orchestrator import _run_agent_from_snapshot
        from agentcore.processing.process import run_graph_internal

        job_id = job_data["job_id"]

        agent_text, _result_sid, was_interrupted, agent_content_blocks = await _run_agent_from_snapshot(
            agent_id=job_data["agent_id"],
            agent_name=job_data["agent_name"],
            snapshot=job_data["snapshot"],
            input_value=job_data["input_value"],
            session_id=job_data["session_id"],
            user_id=job_data["user_id"],
            files=job_data.get("files"),
            stream=True,
            event_manager=event_manager,
            deployment_id=job_data.get("deployment_id"),
            org_id=job_data.get("org_id"),
            dept_id=job_data.get("dept_id"),
            is_prod_deployment=job_data.get("is_prod_deployment", False),
            project_id=job_data.get("project_id"),
            project_name=job_data.get("project_name"),
        )

        if was_interrupted:
            event_manager.on_end(data={})
            await event_manager.queue.put((None, None, time.time()))
            return

        if not agent_text or not agent_text.strip():
            agent_text = "Agent did not produce a response."

        event_manager.on_end(data={"agent_text": agent_text})
        # Sentinel to signal the streaming response consumer to stop
        await event_manager.queue.put((None, None, time.time()))
