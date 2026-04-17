import type {
  LiteSynphoniaProviderSettings,
  SaveLiteSynphoniaProviderSettingsPayload,
} from "../types";

export async function getLiteSynphoniaProviderSettings(): Promise<LiteSynphoniaProviderSettings> {
  if (!window.desktopBridge?.getLiteSynphoniaProviderSettings) {
    throw new Error("当前环境不支持读取 LiteSynphonia 配置。");
  }

  const result = await window.desktopBridge.getLiteSynphoniaProviderSettings();
  if (!result.ok || !result.settings) {
    throw new Error(result.message || "读取 LiteSynphonia 配置失败。");
  }

  return result.settings;
}

export async function saveLiteSynphoniaProviderSettings(
  payload: SaveLiteSynphoniaProviderSettingsPayload,
): Promise<LiteSynphoniaProviderSettings> {
  if (!window.desktopBridge?.saveLiteSynphoniaProviderSettings) {
    throw new Error("当前环境不支持保存 LiteSynphonia 配置。");
  }

  const result = await window.desktopBridge.saveLiteSynphoniaProviderSettings(payload);
  if (!result.ok || !result.settings) {
    throw new Error(result.message || "保存 LiteSynphonia 配置失败。");
  }

  return result.settings;
}
