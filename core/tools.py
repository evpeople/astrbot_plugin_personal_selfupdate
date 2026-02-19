from typing import Optional
from astrbot.api import FunctionTool
from astrbot.api.event import AstrMessageEvent
from astrbot.api import logger
import json
from datetime import datetime
from pathlib import Path

# Forward declaration for type hinting
if False:
    from ..main import Main


def create_get_persona_detail_tool(main_plugin: "Main", event: "AstrMessageEvent") -> FunctionTool:
    async def handler(**kwargs):
        persona_id = kwargs.get('persona_id')
        logger.info(f"[Tool] GetPersonaDetailTool: 查询人格 '{persona_id}' 的详细信息")

        cached_persona = main_plugin.get_cached_persona_detail(persona_id)
        if cached_persona is not None:
            logger.info(f"[Tool] GetPersonaDetailTool: 使用缓存的人格 '{persona_id}' 信息")
            return json.dumps({"ok": True, "persona": cached_persona}, ensure_ascii=False)

        try:
            persona = await main_plugin.context.persona_manager.get_persona(persona_id)
            if not persona:
                raise ValueError("未找到指定人格")
            logger.info(f"[Tool] GetPersonaDetailTool: 成功获取人格 '{persona_id}' 信息")
            result = {
                "persona_id": persona_id,
                "system_prompt": getattr(persona, "system_prompt", ""),
                "begin_dialogs": getattr(persona, "begin_dialogs", []),
                "tools": getattr(persona, "tools", None)
            }
            main_plugin.cache_persona_detail(persona_id, result)
            return json.dumps({"ok": True, "persona": result}, ensure_ascii=False)
        except Exception as e:
            logger.error(f"[Tool] GetPersonaDetailTool: 获取人格 '{persona_id}' 失败: {e}")
            return json.dumps({"ok": False, "error": str(e), "persona_id": persona_id}, ensure_ascii=False)

    return FunctionTool(
        name="get_persona_detail",
        description="获取指定ID的人格的详细信息。",
        parameters={
            "type": "object",
            "properties": {
                "persona_id": {
                    "type": "string",
                    "description": "要查询的人格的ID。"
                }
            },
            "required": ["persona_id"]
        },
        handler=handler,
    )


def _save_persona_backup(plugin: "Main", persona_id: str, data: dict) -> None:
    """将旧人格数据保存为 JSON 备份文件，并根据 max_backups 配置清理旧备份。"""
    backup_dir: Path = plugin._backup_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = backup_dir / f"{persona_id}_{timestamp}.json"

    backup_data = {**data, "backup_time": datetime.now().isoformat()}
    backup_file.write_text(json.dumps(backup_data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"[Backup] 已备份人格 '{persona_id}' -> {backup_file.name}")

    # 根据配置清理旧备份
    max_backups = int(plugin.config.get("max_backups", 10))
    if max_backups == -1:
        return

    existing = sorted(backup_dir.glob(f"{persona_id}_*.json"), key=lambda f: f.stat().st_mtime)
    while len(existing) > max_backups:
        oldest = existing.pop(0)
        oldest.unlink()
        logger.info(f"[Backup] 已清理旧备份 {oldest.name}")


def create_update_persona_details_tool(main_plugin: "Main", event: "AstrMessageEvent") -> FunctionTool:
    async def handler(**kwargs):
        persona_id = kwargs.get('persona_id')
        system_prompt = kwargs.get('system_prompt')
        begin_dialogs = kwargs.get('begin_dialogs')
        tools = kwargs.get('tools')

        logger.info(f"[Tool] UpdatePersonaDetailsTool: 更新人格 '{persona_id}' - system_prompt: {bool(system_prompt)}, begin_dialogs: {bool(begin_dialogs)}, tools: {bool(tools)}")

        # 备份旧人格数据
        try:
            old_data = main_plugin.get_cached_persona_detail(persona_id)
            if old_data is None:
                old_persona = await main_plugin.context.persona_manager.get_persona(persona_id)
                if old_persona:
                    old_data = {
                        "persona_id": persona_id,
                        "system_prompt": getattr(old_persona, "system_prompt", ""),
                        "begin_dialogs": getattr(old_persona, "begin_dialogs", []),
                        "tools": getattr(old_persona, "tools", None),
                    }
            if old_data is not None:
                _save_persona_backup(main_plugin, persona_id, old_data)
        except Exception as e:
            logger.warning(f"[Tool] UpdatePersonaDetailsTool: 备份人格 '{persona_id}' 失败，继续更新: {e}")

        if begin_dialogs is not None:
            if not isinstance(begin_dialogs, list) or any(not isinstance(item, str) for item in begin_dialogs):
                error_msg = "begin_dialogs 必须是字符串列表"
                logger.warning(f"[Tool] UpdatePersonaDetailsTool: {error_msg}")
                return json.dumps({"ok": False, "error": error_msg}, ensure_ascii=False)

            if len(begin_dialogs) % 2 != 0:
                error_msg = "begin_dialogs 条目数量必须为偶数，且应按用户/助手顺序排列"
                logger.warning(f"[Tool] UpdatePersonaDetailsTool: {error_msg}")
                return json.dumps({"ok": False, "error": error_msg}, ensure_ascii=False)

        try:
            persona = await main_plugin.context.persona_manager.update_persona(
                persona_id,
                system_prompt=system_prompt,
                begin_dialogs=begin_dialogs,
                tools=tools,
            )
            logger.info(f"[Tool] UpdatePersonaDetailsTool: 成功更新人格 '{persona_id}'")
        except Exception as e:
            logger.error(f"[Tool] UpdatePersonaDetailsTool: 更新人格 '{persona_id}' 失败: {e}")
            return json.dumps({"ok": False, "error": f"更新失败：{e}"}, ensure_ascii=False)

        try:
            persona = persona or await main_plugin.context.persona_manager.get_persona(persona_id)
            if not persona:
                raise ValueError("更新后无法获取人格详情")
            result = {
                "persona_id": persona_id,
                "system_prompt": getattr(persona, "system_prompt", ""),
                "begin_dialogs": getattr(persona, "begin_dialogs", []),
                "tools": getattr(persona, "tools", None)
            }
            logger.info(f"[Tool] UpdatePersonaDetailsTool: 返回更新后的人格信息")
            main_plugin.cache_persona_detail(persona_id, result)
            return json.dumps({"ok": True, "persona": result}, ensure_ascii=False)
        except Exception as e:
            logger.error(f"[Tool] UpdatePersonaDetailsTool: 获取更新后信息失败: {e}")
            return json.dumps({"ok": False, "error": f"更新成功但获取更新后信息失败：{e}"}, ensure_ascii=False)

    return FunctionTool(
        name="update_persona_details",
        description="更新指定ID的人格信息。只有提供的参数才会被更新。",
        parameters={
            "type": "object",
            "properties": {
                "persona_id": {
                    "type": "string",
                    "description": "要更新的人格的ID。"
                },
                "system_prompt": {
                    "type": "string",
                    "description": "新的系统提示。"
                },
                "begin_dialogs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "新的开场白列表。"
                },
                "tools": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "新的工具列表。None表示默认全部，空列表表示无。"
                }
            },
            "required": ["persona_id"]
        },
        handler=handler,
    )
