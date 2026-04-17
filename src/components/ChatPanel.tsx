import { useEffect, useRef, type FormEvent, type KeyboardEvent } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { ChatMessage } from "../types";

interface ChatPanelProps {
  messages: ReadonlyArray<ChatMessage>;
  value: string;
  isAsking: boolean;
  onChange: (nextValue: string) => void;
  onSubmit: () => void;
}

export default function ChatPanel({
  messages,
  value,
  isAsking,
  onChange,
  onSubmit,
}: ChatPanelProps) {
  const messagesRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!messagesRef.current) {
      return;
    }

    messagesRef.current.scrollTop = messagesRef.current.scrollHeight;
  }, [messages, isAsking]);

  function handleSubmit(event: FormEvent<HTMLFormElement>): void {
    event.preventDefault();
    onSubmit();
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>): void {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      onSubmit();
    }
  }

  return (
    <>
      <div className="chat-header">
        <h2 className="section-title">智能对话</h2>
      </div>

      <div className="chat-shell">
        <div className="chat-messages" ref={messagesRef}>
          {!messages.length ? (
            <div className="chat-empty">
              <div className="chat-empty-title">开始对话</div>
              <div className="chat-empty-copy">
                不需要先打开文件；如果当前有工作区，回答会自动参考课件内容。
              </div>
            </div>
          ) : null}
          {messages.map((message, index) => (
            <div className={`message ${message.role}`} key={`${message.role}-${index}`}>
              {message.role === "assistant" ? (
                <div className="message-copy message-copy-markdown">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {message.text}
                  </ReactMarkdown>
                </div>
              ) : (
                <div className="message-copy">{message.text}</div>
              )}
              {message.citations?.length ? (
                <div className="message-citations">
                  {message.citations.map((citation) => (
                    <span className="message-citation" key={citation}>
                      {citation}
                    </span>
                  ))}
                </div>
              ) : null}
            </div>
          ))}
          {isAsking ? <div className="message assistant">AI 正在思考中...</div> : null}
        </div>

        <form className="chat-form" onSubmit={handleSubmit}>
          <div className="chat-input-shell">
            <textarea
              value={value}
              onChange={(event) => onChange(event.target.value)}
              placeholder="输入消息"
              onKeyDown={handleKeyDown}
            />
            <button
              className="secondary-button chat-submit-button"
              type="submit"
              disabled={isAsking || !value.trim()}
            >
              发送
            </button>
          </div>
        </form>
      </div>
    </>
  );
}
