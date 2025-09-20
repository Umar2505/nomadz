"use client";

import Link from "next/link";
import { useState } from "react";

type Message = {
  id: string;
  role: "user" | "assistant";
  name: string;
  text: string;
  timestamp: string;
};

type HistoryItem = {
  id: string;
  title: string;
  preview: string;
  updatedAt: string;
};

const DEFAULT_MESSAGES: Message[] = [
  {
    id: "welcome",
    role: "assistant",
    name: "Nomadz AI",
    text: "Hey traveler! I&apos;m here to design flexible remote-work adventures tailored to your vibe. Where do you want to explore next?",
    timestamp: "09:00",
  },
];

const PREVIOUS_CHATS: HistoryItem[] = [
  {
    id: "tokyo-itinerary",
    title: "Tokyo coworking sprint",
    preview: "7-day blend of neon nights and Zen mornings",
    updatedAt: "Nov 21",
  },
  {
    id: "lisbon-waves",
    title: "Lisbon surf & workweek",
    preview: "Cowork-friendly cafÃ©s near praia do Guincho",
    updatedAt: "Nov 18",
  },
  {
    id: "seoul-seasonal",
    title: "Seoul seasonal eats",
    preview: "Street food crawl with late-night work hubs",
    updatedAt: "Nov 12",
  },
];

const timeFormatter = new Intl.DateTimeFormat([], {
  hour: "numeric",
  minute: "2-digit",
});

const FALLBACK_ASSISTANT_RESPONSE =
  "Nomadz AI connected but didn&apos;t return any details. Try asking again.";

function extractAssistantText(payload: unknown): string {
  if (payload && typeof payload === "object") {
    const { output, response, message } = payload as {
      output?: unknown;
      response?: unknown;
      message?: unknown;
    };

    const candidates = [output, response, message];

    for (const candidate of candidates) {
      if (typeof candidate === "string" && candidate.trim().length > 0) {
        return candidate;
      }
    }
  }

  return FALLBACK_ASSISTANT_RESPONSE;
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>(() =>
    DEFAULT_MESSAGES.map((message) => ({ ...message }))
  );
  const [inputValue, setInputValue] = useState("");
  const [showIntro, setShowIntro] = useState(true);
  const [isLoading, setIsLoading] = useState(false);

  const handleSend = async (event?: React.FormEvent<HTMLFormElement>) => {
    event?.preventDefault();

    const trimmed = inputValue.trim();

    if (!trimmed || isLoading) {
      return;
    }

    const now = new Date();
    const userMessage: Message = {
      id: `user-${now.getTime()}`,
      role: "user",
      name: "You",
      text: trimmed,
      timestamp: timeFormatter.format(now),
    };

    const assistantMessageId = `assistant-${now.getTime() + 1}`;
    const assistantMessage: Message = {
      id: assistantMessageId,
      role: "assistant",
      name: "Nomadz AI",
      text: "Nomadz AI is mapping your remote-work adventure...",
      timestamp: timeFormatter.format(now),
    };

    setMessages((current) => [...current, userMessage, assistantMessage]);
    setInputValue("");
    setShowIntro(false);
    setIsLoading(true);

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query: trimmed }),
      });

      let payload: unknown = null;

      try {
        payload = await response.json();
      } catch {
        payload = null;
      }

      if (!response.ok) {
        const errorMessage =
          payload &&
          typeof payload === "object" &&
          "error" in payload &&
          typeof (payload as { error?: unknown }).error === "string"
            ? (payload as { error: string }).error
            : "The Nomadz API returned an unexpected error.";

        throw new Error(errorMessage);
      }

      const assistantText = extractAssistantText(payload);

      setMessages((current) =>
        current.map((message) =>
          message.id === assistantMessageId
            ? {
                ...message,
                text: assistantText,
                timestamp: timeFormatter.format(new Date()),
              }
            : message,
        ),
      );
    } catch (error) {
      const fallbackError =
        error instanceof Error
          ? error.message
          : "Nomadz AI encountered an unexpected issue.";

      setMessages((current) =>
        current.map((message) =>
          message.id === assistantMessageId
            ? {
                ...message,
                text: `Nomadz AI ran into an issue: ${fallbackError}`,
                timestamp: timeFormatter.format(new Date()),
              }
            : message,
        ),
      );
    } finally {
      setIsLoading(false);
    }
  };

  const handleNewChat = () => {
    if (isLoading) {
      return;
    }

    setMessages(DEFAULT_MESSAGES.map((message) => ({ ...message })));
    setInputValue("");
    setShowIntro(true);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-indigo-950 to-purple-900 text-slate-100">
      <div className="flex min-h-screen flex-col bg-black/20 backdrop-blur-xl">
        <header className="border-b border-white/10 bg-black/40">
          <div className="mx-auto flex w-full max-w-6xl items-center justify-between px-6 py-6">
            <div className="flex items-center gap-4">
              <span className="inline-flex h-12 w-12 items-center justify-center rounded-full bg-gradient-to-tr from-indigo-500 to-purple-400 text-lg font-semibold text-white shadow-lg">
                N
              </span>
              <div>
                <p className="text-lg font-semibold leading-tight">Nomadz Compass</p>
                <p className="text-xs text-white/60">
                  Plan remote work adventures with an AI travel copilot.
                </p>
              </div>
            </div>
            <Link
              href="/sign-in"
              className="rounded-full bg-white/10 px-5 py-2.5 text-sm font-medium text-white shadow-lg transition hover:bg-white/20 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white"
            >
              Sign in
            </Link>
          </div>
        </header>

        <div className="flex flex-1 overflow-hidden">
          <aside className="hidden w-72 flex-col border-r border-white/10 bg-white/5 px-6 py-8 backdrop-blur-2xl lg:flex">
            <div className="mb-6 flex items-center justify-between">
              <h2 className="text-xs font-semibold uppercase tracking-[0.3em] text-white/70">
                Recent chats
              </h2>
              <span className="h-2 w-2 rounded-full bg-emerald-400 shadow-[0_0_12px_2px_rgba(52,211,153,0.6)]" />
            </div>
            <nav className="flex-1 space-y-3 overflow-y-auto pr-2">
              {PREVIOUS_CHATS.map((chat) => (
                <button
                  key={chat.id}
                  type="button"
                  className="w-full rounded-2xl border border-white/10 bg-black/30 px-4 py-4 text-left transition hover:border-white/30 hover:bg-white/10"
                >
                  <p className="text-sm font-semibold text-white">
                    {chat.title}
                  </p>
                  <p className="mt-1 text-xs text-white/60">{chat.preview}</p>
                  <p className="mt-3 text-[11px] uppercase tracking-[0.2em] text-white/40">
                    Updated {chat.updatedAt}
                  </p>
                </button>
              ))}
            </nav>
          </aside>

          <main className="flex flex-1 flex-col overflow-hidden">
            <div className="mx-auto flex h-full w-full max-w-4xl flex-col gap-8 px-6 py-10">
              <section className="min-h-[64px]">
                {showIntro ? (
                  <p className="max-w-2xl text-sm text-white/80">
                    Nomadz Compass transforms scattered travel research into a single actionable plan so you can land, plug in, and start living like a local from day one.
                  </p>
                ) : (
                  <div className="flex items-center gap-3 rounded-3xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-white/70 shadow-lg">
                    <span className="inline-flex h-8 w-8 items-center justify-center rounded-full bg-gradient-to-tr from-indigo-500 to-sky-400 text-xs font-semibold uppercase text-white">
                      Beta
                    </span>
                    Responses are streaming directly from the Nomadz intelligence engine via the connected API.
                  </div>
                )}
              </section>

              <section className="flex flex-1 flex-col overflow-hidden rounded-3xl border border-white/10 bg-black/40 shadow-2xl">
                <div className="flex items-center justify-between border-b border-white/10 px-6 py-5">
                  <div>
                    <p className="text-xs font-semibold uppercase tracking-[0.3em] text-white/60">
                      Live conversation
                    </p>
                    <p className="mt-1 text-sm text-white/70">
                      You&apos;re chatting with Nomadz AI.
                    </p>
                  </div>
                </div>

                <div className="flex-1 space-y-4 overflow-y-auto px-6 py-6">
                  {messages.map((message) => (
                    <div
                      key={message.id}
                      className={`flex ${
                        message.role === "user" ? "justify-end" : "justify-start"
                      }`}
                    >
                      <div
                        className={`max-w-[75%] rounded-3xl px-5 py-4 text-sm leading-6 shadow-xl backdrop-blur ${
                          message.role === "user"
                            ? "bg-gradient-to-br from-indigo-500 to-purple-500 text-white"
                            : "bg-white/10 text-white/90"
                        }`}
                      >
                        <div className="flex items-center justify-between text-[11px] uppercase tracking-[0.2em] text-white/60">
                          <span>{message.name}</span>
                          <span>{message.timestamp}</span>
                        </div>
                        <p className="mt-3 whitespace-pre-line text-sm text-white/90">
                          {message.text}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>

                <form
                  onSubmit={handleSend}
                  className="border-t border-white/10 bg-black/60 px-6 py-5"
                >
                  <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
                    <button
                      type="button"
                      onClick={handleNewChat}
                      disabled={isLoading}
                      className="flex h-12 w-full items-center justify-center rounded-2xl border border-dashed border-white/20 bg-transparent text-sm font-medium text-white/80 transition hover:border-white/50 hover:text-white focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white disabled:cursor-not-allowed disabled:opacity-60 sm:w-12"
                    >
                      <svg
                        aria-hidden="true"
                        className="h-5 w-5"
                        viewBox="0 0 20 20"
                        fill="none"
                        xmlns="http://www.w3.org/2000/svg"
                      >
                        <path
                          d="M10 4v12M4 10h12"
                          stroke="currentColor"
                          strokeWidth="1.6"
                          strokeLinecap="round"
                        />
                      </svg>
                      <span className="sr-only">Start a new chat</span>
                    </button>
                    <div className="flex flex-1 items-center gap-3 rounded-2xl border border-white/10 bg-white/10 px-4 py-3 transition focus-within:border-indigo-400 focus-within:ring-2 focus-within:ring-indigo-500/50">
                      <input
                        value={inputValue}
                        onChange={(event) => setInputValue(event.target.value)}
                        placeholder="Ask Nomadz AI anything about your next remote work trip..."
                        className="flex-1 bg-transparent text-sm text-white placeholder:text-white/50 focus:outline-none"
                        aria-label="Type your message"
                      />
                      <button
                        type="submit"
                        disabled={isLoading}
                        className="flex h-10 w-10 items-center justify-center rounded-full bg-gradient-to-tr from-sky-500 to-indigo-500 text-white shadow-lg transition hover:from-sky-400 hover:to-indigo-400 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white disabled:cursor-not-allowed disabled:opacity-60"
                        aria-label="Send message"
                      >
                        <svg
                          aria-hidden="true"
                          className="h-4 w-4"
                          viewBox="0 0 20 20"
                          fill="none"
                          xmlns="http://www.w3.org/2000/svg"
                        >
                          <path
                            d="M3.227 9.38 15.06 3.575c.87-.43 1.786.486 1.357 1.357L10.61 16.764c-.448.907-1.804.81-2.065-.146l-1.225-4.446a.5.5 0 0 0-.347-.347L2.527 10.6c-.956-.26-1.054-1.617-.147-2.065z"
                            fill="currentColor"
                          />
                        </svg>
                      </button>
                    </div>
                  </div>
                </form>
              </section>
            </div>
          </main>
        </div>
      </div>
    </div>
  );
}

