from typing import Optional
from pathlib import Path

from astrbot.api.star import Star, Context, register
from astrbot.api.event import filter, AstrMessageEvent, MessageEventResult
from astrbot.api.provider import LLMResponse
from astrbot.api import AstrBotConfig, logger, ToolSet
from astrbot.core.utils.astrbot_path import get_astrbot_data_path

from .core.tools import create_get_persona_detail_tool, create_update_persona_details_tool

import json

SYSTEM_PROMPT_TEMPLATE = """你是人格配置专家，负责根据用户要求更新 AI 人格设定。
可用工具：
- get_persona_detail(persona_id): 获取人格当前设定 - 必须先调用
- update_persona_details(persona_id, system_prompt?, begin_dialogs?, tools?): 更新人格设定,begin_dialogs为偶数个字符串，每个字符串代表一个对话，用户和助手轮流对话

任务：更新人格 '{persona_id}'，要求：{update_requirement}

重要：你必须严格按以下步骤执行：
1. 调用 get_persona_detail 获取当前人格信息
2. 根据要求分析需要修改的内容  
3. 调用 update_persona_details 应用修改
4. 简洁总结修改内容

请严格按照上述流程执行。特别注意：
- begin_dialogs 必须包含偶数条对话，且需按照"用户、助手"轮流排列。
- 只有在完成分析并确定改动后，才调用一次 update_persona_details 应用修改。

以下是该对话窗口最近的聊天记录，供你参考用户的对话风格和使用场景：
{chat_history_text}

完成所有步骤后，请以 '{completion_sentinel}' 开头提供最终总结，简要说明修改内容及影响。

请立即开始执行，先调用 get_persona_detail 工具。"""

DEFAULT_USER_PROMPT = "开始执行。"
TOOL_CALL_PLACEHOLDER_PROMPT = " "
MAX_AGENT_ITERATIONS = 10
COMPLETION_SENTINEL = "[AGENT_DONE]"  # Agent completion marker


class ProviderResolutionError(Exception):
    """Raised when an active provider cannot be resolved."""


class AgentExecutionError(Exception):
    """Raised when the agent loop cannot complete successfully."""

@register(
    "personal_selfupdate",
    "kterna",
    "通过与LLM对话来更新人格",
    "0.1.1",
    "https://github.com/kterna/astrbot_plugin_personal_selfupdate"
)
class Main(Star):
    def __init__(self, context: Context, config: AstrBotConfig) -> None:
        """
        插件初始化
        """
        super().__init__(context)
        self.config = config
        self._persona_cache = {}

        # 备份目录
        self._backup_dir: Path = get_astrbot_data_path() / "plugin_data" / "personal_selfupdate" / "backups"
        self._backup_dir.mkdir(parents=True, exist_ok=True)

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("人格更新", alias={"persona_update"})
    async def persona_self_update(self, event: AstrMessageEvent):
        """
        通过独立的Agent流程，让LLM自我更新人格。
        用法: /人格更新 [人格ID] [更新要求]
        例如: /人格更新 伯特 让他说话更专业一些
        """
        try:
            persona_id, update_requirement = self._parse_update_request(event)
        except ValueError as error:
            yield event.plain_result(str(error))
            return

        self._reset_persona_cache()

        logger.info(f"收到人格更新命令. ID: '{persona_id}', 要求: '{update_requirement}'")

        tool_set = self._build_tool_set(event)

        try:
            provider, model_name = self._resolve_provider(event)
        except ProviderResolutionError as error:
            yield event.plain_result(f"获取服务提供商失败: {error}")
            return

        # 不获取聊天记录，保持原有行为
        system_prompt = self._build_system_prompt(persona_id, update_requirement, chat_history=None)
        user_prompt = self._initial_user_prompt()

        logger.info("开始调用 LLM Agent 进行人格更新")
        yield event.plain_result("🔄 分析中...")

        try:
            final_text = await self._run_agent_conversation(
                provider=provider,
                model_name=model_name,
                tool_set=tool_set,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            yield event.plain_result(f"✅ 更新完成\n{final_text}")
        except AgentExecutionError as error:
            logger.error(f"执行人格更新 Agent 流程时出错: {error}", exc_info=True)
            yield event.plain_result(f"❌ 更新失败: {error}")
        except Exception as error:
            logger.error(f"执行人格更新 Agent 流程时出错: {error}", exc_info=True)
            yield event.plain_result(f"❌ 更新失败: {error}")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("人格更新高级", alias={"persona_update_advanced"})
    async def persona_self_update_advanced(self, event: AstrMessageEvent):
        """
        通过独立的Agent流程，让LLM自我更新人格，支持指定使用多少条聊天记录。
        用法: /人格更新高级 [人格ID] [消息数量/-1] [更新要求]
        例如: /人格更新高级 伯特 -1 让他说话更专业一些（使用所有聊天记录）
        例如: /人格更新高级 伯特 10 让他说话更专业一些（使用最近10条消息）
        """
        try:
            persona_id, message_count, update_requirement = self._parse_advanced_update_request(event)
        except ValueError as error:
            yield event.plain_result(str(error))
            return

        self._reset_persona_cache()

        logger.info(f"收到人格更新命令(高级). ID: '{persona_id}', 消息数量: {message_count}, 要求: '{update_requirement}'")

        tool_set = self._build_tool_set(event)

        try:
            provider, model_name = self._resolve_provider(event)
        except ProviderResolutionError as error:
            yield event.plain_result(f"获取服务提供商失败: {error}")
            return

        # 获取当前会话的聊天记录，支持指定数量
        chat_history = await self._get_chat_history(event, message_count)

        system_prompt = self._build_system_prompt(persona_id, update_requirement, chat_history)
        user_prompt = self._initial_user_prompt()

        logger.info("开始调用 LLM Agent 进行人格更新")
        yield event.plain_result("🔄 分析中...")

        try:
            final_text = await self._run_agent_conversation(
                provider=provider,
                model_name=model_name,
                tool_set=tool_set,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            yield event.plain_result(f"✅ 更新完成\n{final_text}")
        except AgentExecutionError as error:
            logger.error(f"执行人格更新 Agent 流程时出错: {error}", exc_info=True)
            yield event.plain_result(f"❌ 更新失败: {error}")
        except Exception as error:
            logger.error(f"执行人格更新 Agent 流程时出错: {error}", exc_info=True)
            yield event.plain_result(f"❌ 更新失败: {error}")

    def _parse_update_request(self, event: AstrMessageEvent) -> tuple[str, str]:
        raw_message = event.message_str.strip()
        parts = raw_message.split(None, 2) if raw_message else []

        if len(parts) < 3:
            raise ValueError("参数不足，请提供人格ID和更新要求。")

        _, persona_id, update_requirement = parts
        persona_id = persona_id.strip()
        update_requirement = update_requirement.strip()

        if not persona_id:
            raise ValueError("人格ID 不能为空，请重新输入。")

        if not update_requirement:
            raise ValueError("更新要求不能为空，请提供具体说明。")

        return persona_id, update_requirement

    def _parse_advanced_update_request(
        self,
        event: AstrMessageEvent
    ) -> tuple[str, int, str]:
        """解析高级人格更新请求。

        Returns:
            tuple: (persona_id, message_count, update_requirement)
        """
        raw_message = event.message_str.strip()
        parts = raw_message.split(None, 3) if raw_message else []

        if len(parts) < 4:
            raise ValueError(
                "参数不足，请提供人格ID、消息数量和更新要求。\n"
                "用法: /人格更新高级 [人格ID] [消息数量/-1] [更新要求]\n"
                "示例: /人格更新高级 伯特 -1 让他说话更专业一些\n"
                "      /人格更新高级 伯特 10 让他说话更专业一些"
            )

        _, persona_id, message_count_str, update_requirement = parts
        persona_id = persona_id.strip()
        update_requirement = update_requirement.strip()

        if not persona_id:
            raise ValueError("人格ID 不能为空，请重新输入。")

        # 解析消息数量
        try:
            message_count = int(message_count_str)
            if message_count != -1 and message_count < 1:
                raise ValueError("消息数量必须大于0，或使用-1表示所有记录。")
        except ValueError:
            raise ValueError(f"消息数量无效，请输入整数。-1表示所有记录，其他正整数表示最近多少条消息。")

        if not update_requirement:
            raise ValueError("更新要求不能为空，请提供具体说明。")

        return persona_id, message_count, update_requirement

    async def _get_chat_history(
        self,
        event: AstrMessageEvent,
        message_count: int | None = None
    ) -> list[dict]:
        """获取当前会话的聊天记录。

        Args:
            event: 消息事件
            message_count: 消息数量，None 表示返回所有记录（由 _format_chat_history 截取），-1 表示所有记录，正整数表示最近多少条
        """
        try:
            conv_mgr = self.context.conversation_manager
            umo = event.unified_msg_origin
            cid = await conv_mgr.get_curr_conversation_id(umo)
            if cid:
                conversation = await conv_mgr.get_conversation(umo, cid)
                if conversation and conversation.history:
                    history = json.loads(conversation.history)
                    # 根据 message_count 截取历史记录
                    if message_count is None or message_count == -1:
                        # None 或 -1：返回所有记录
                        return history
                    else:
                        # 正整数：返回最近 N 条记录
                        return history[-message_count:] if len(history) > message_count else history
            return []
        except Exception as error:
            logger.warning(f"获取聊天记录失败: {error}")
            return []

    def _build_tool_set(self, event: AstrMessageEvent) -> ToolSet:
        return ToolSet([
            create_get_persona_detail_tool(main_plugin=self, event=event),
            create_update_persona_details_tool(main_plugin=self, event=event),
        ])

    def _resolve_provider(self, event: AstrMessageEvent) -> tuple[object, Optional[str]]:
        provider_id = str(self.config.get("provider", "") or "").strip()
        model_name = self.config.get("model", "")
        model_name = str(model_name) if model_name and model_name != "" else None

        try:
            provider_instance = None

            if provider_id:
                provider_instance = self.context.get_provider_by_id(provider_id=provider_id)
                if not provider_instance:
                    logger.warning(f"指定的 Provider '{provider_id}' 不存在或未启用，使用默认 provider")
                    provider_instance = self.context.get_using_provider(umo=event.unified_msg_origin)
            else:
                provider_instance = self.context.get_using_provider(umo=event.unified_msg_origin)

        except Exception as error:
            logger.error(f"获取服务提供商失败: {error}", exc_info=True)
            raise ProviderResolutionError(str(error)) from error

        if not provider_instance:
            message = "无法获取有效的服务提供商。请检查是否有启用的 Provider。"
            logger.error(f"获取服务提供商失败: {message}")
            raise ProviderResolutionError(message)

        return provider_instance, model_name

    def _build_system_prompt(
        self,
        persona_id: str,
        update_requirement: str,
        chat_history: list[dict] | None = None
    ) -> str:
        chat_history_text = self._format_chat_history(chat_history)
        return SYSTEM_PROMPT_TEMPLATE.format(
            persona_id=persona_id,
            update_requirement=update_requirement,
            completion_sentinel=COMPLETION_SENTINEL,
            chat_history_text=chat_history_text,
        )

    def _format_chat_history(self, chat_history: list[dict] | None) -> str:
        """将聊天记录格式化为可读文本，截取最近20条消息。"""
        if chat_history is None or len(chat_history) == 0:
            return "（无历史聊天记录）"

        # 截取最近 20 条消息
        recent_history = chat_history[-20:] if len(chat_history) > 20 else chat_history

        lines = []
        for msg in recent_history:
            role = msg.get("role", "unknown")
            content = msg.get("content") or ""
            # 截断过长的内容
            if len(content) > 500:
                content = content[:500] + "..."
            lines.append(f"[{role}]: {content}")

        return "\n".join(lines) if lines else "（无历史聊天记录）"

    def _initial_user_prompt(self) -> str:
        return DEFAULT_USER_PROMPT

    async def _run_agent_conversation(
        self,
        provider,
        model_name: Optional[str],
        tool_set: ToolSet,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        logger.info("开始 LLM Agent 工具调用...")

        max_iterations = MAX_AGENT_ITERATIONS
        messages: list[dict] = []
        current_prompt = user_prompt
        final_text = ""

        try:
            for _ in range(max_iterations):
                response: LLMResponse = await provider.text_chat(
                    prompt=current_prompt,
                    system_prompt=system_prompt,
                    model=model_name or None,
                    func_tool=tool_set,
                    session_id=None,
                    contexts=messages,
                    image_urls=[]
                )

                messages.append({"role": "user", "content": current_prompt})

                if (hasattr(response, 'tools_call_name') and
                    hasattr(response, 'tools_call_args') and
                    response.tools_call_name and
                    response.tools_call_args):

                    tool_results = []
                    tool_call_ids = getattr(
                        response,
                        'tools_call_ids',
                        [f"{name}:{i}" for i, name in enumerate(response.tools_call_name)]
                    )

                    for tool_name, tool_args, tool_id in zip(
                        response.tools_call_name,
                        response.tools_call_args,
                        tool_call_ids
                    ):
                        tool = tool_set.get_tool(tool_name)
                        if tool and tool.handler:
                            try:
                                result = await tool.handler(**tool_args)
                                tool_results.append({
                                    "tool_call_id": tool_id,
                                    "role": "tool",
                                    "content": str(result)
                                })
                            except Exception as error:
                                logger.error(f"工具 {tool_name} 执行失败: {error}")
                                tool_results.append({
                                    "tool_call_id": tool_id,
                                    "role": "tool",
                                    "content": f"工具执行失败: {error}"
                                })
                        else:
                            logger.error(f"未找到工具: {tool_name}")
                            tool_results.append({
                                "tool_call_id": tool_id,
                                "role": "tool",
                                "content": f"未找到工具: {tool_name}"
                            })

                    assistant_content = "调用工具"
                    if response.result_chain and response.result_chain.chain:
                        text = response.result_chain.chain[0].text
                        if text and text.strip():
                            assistant_content = text

                    messages.append({
                        "role": "assistant",
                        "content": assistant_content,
                        "tool_calls": [
                            {
                                "id": tool_id,
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": json.dumps(tool_args) if isinstance(tool_args, dict) else str(tool_args)
                                }
                            }
                            for tool_name, tool_args, tool_id in zip(
                                response.tools_call_name,
                                response.tools_call_args,
                                tool_call_ids
                            )
                        ]
                    })

                    messages.extend(tool_results)
                    current_prompt = TOOL_CALL_PLACEHOLDER_PROMPT
                    continue

                final_text = response.completion_text if hasattr(response, 'completion_text') else ""
                if response.result_chain and response.result_chain.chain:
                    final_text = response.result_chain.chain[0].text

                break
            else:
                final_text = "工具调用超过最大次数限制"
                logger.warning("工具调用循环达到最大次数限制")
        except Exception as error:
            raise AgentExecutionError(str(error)) from error

        return self._extract_completion_text(final_text)

    def _reset_persona_cache(self) -> None:
        """Clear per-command persona cache so each invocation starts fresh."""
        self._persona_cache.clear()

    def get_cached_persona_detail(self, persona_id: str) -> Optional[dict]:
        return self._persona_cache.get(persona_id)

    def cache_persona_detail(self, persona_id: str, detail: dict) -> None:
        self._persona_cache[persona_id] = detail

    def _extract_completion_text(self, raw_text: str) -> str:
        if not raw_text:
            return raw_text

        text = raw_text.strip()

        if COMPLETION_SENTINEL in text:
            _, remainder = text.split(COMPLETION_SENTINEL, 1)
            remainder = remainder.strip()
            return remainder if remainder else COMPLETION_SENTINEL

        return text
