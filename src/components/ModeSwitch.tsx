import type { ViewMode } from "../types";

interface ModeSwitchProps {
  value: ViewMode;
  onChange: (nextMode: ViewMode) => void;
}

export default function ModeSwitch({
  value,
  onChange,
}: ModeSwitchProps) {
  return (
    <div
      className={`mode-switch mode-${value}`}
      role="tablist"
      aria-label="界面模式"
    >
      <span className="mode-switch-thumb" aria-hidden="true" />
      <button
        className={`mode-switch-button ${value === "normal" ? "active" : ""}`}
        type="button"
        role="tab"
        aria-selected={value === "normal"}
        onClick={() => onChange("normal")}
      >
        Real-time
      </button>
      <button
        className={`mode-switch-button ${value === "review" ? "active" : ""}`}
        type="button"
        role="tab"
        aria-selected={value === "review"}
        onClick={() => onChange("review")}
      >
        Review
      </button>
    </div>
  );
}
