"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  addDoc,
  collection,
  deleteDoc,
  doc,
  onSnapshot,
  orderBy,
  query,
  serverTimestamp,
  setDoc,
  updateDoc,
  type DocumentData,
  type QueryDocumentSnapshot,
} from "firebase/firestore";

import { useAuth } from "@/context/AuthContext";
import { db } from "@/lib/firebase";

type Message = {
  id: string;
  role: "user" | "assistant";
  name: string;
  text: string;
  timestamp: string;
  createdAt: string;
  violations?: string[];
};

type HistoryItem = {
  id: string;
  title: string;
  preview: string;
  updatedAt: string;
  messages: Message[];
};

const timeFormatter = new Intl.DateTimeFormat([], {
  hour: "numeric",
  minute: "2-digit",
});

const historyDateFormatter = new Intl.DateTimeFormat([], {
  month: "short",
  day: "numeric",
});

const FALLBACK_ASSISTANT_RESPONSE =
  "Nomadz AI connected but didn\u2019t return any privacy-filtered insights. Please try again.";

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

function sanitizeViolations(value: unknown): string[] | undefined {
  if (!Array.isArray(value)) {
    return undefined;
  }

  const entries = value as unknown[];

  const sanitizedViolations = entries
    .map((entry) => (typeof entry === "string" ? entry.trim() : null))
    .filter((entry): entry is string => entry !== null && entry.length > 0);

  if (sanitizedViolations.length > 0) {
    return sanitizedViolations;
  }

  if (entries.length === 0 || entries.every((entry) => typeof entry === "string")) {
    return [];
  }

  return undefined;
}

function extractViolations(payload: unknown): string[] | undefined {
  if (!payload || typeof payload !== "object") {
    return undefined;
  }

  const { violations } = payload as { violations?: unknown };

  return sanitizeViolations(violations);
}

function createDefaultMessages(): Message[] {
  const now = new Date();

  return [
    {
      id: "welcome",
      role: "assistant",
      name: "Nomadz AI",
      text: "Hi there! I help you query sensitive data while enforcing privacy safeguards. What question should we investigate?",
      timestamp: timeFormatter.format(now),
      createdAt: now.toISOString(),
    },
  ];
}

function parseMessages(data: unknown): Message[] {
  if (!Array.isArray(data)) {
    return [];
  }

  return data
    .map((entry) => {
      if (!entry || typeof entry !== "object") {
        return null;
      }

      const { id, role, name, text, timestamp, createdAt, violations } =
        entry as Partial<Message> & {
          createdAt?: unknown;
          timestamp?: unknown;
          violations?: unknown;
        };

      if (
        typeof id !== "string" ||
        (role !== "user" && role !== "assistant") ||
        typeof name !== "string" ||
        typeof text !== "string"
      ) {
        return null;
      }

      const createdAtIso =
        typeof createdAt === "string" && !Number.isNaN(Date.parse(createdAt))
          ? createdAt
          : new Date().toISOString();
      const timestampLabel =
        typeof timestamp === "string" && timestamp.trim().length > 0
          ? timestamp
          : timeFormatter.format(new Date(createdAtIso));

      const parsedViolations = sanitizeViolations(violations);

      return {
        id,
        role,
        name,
        text,
        timestamp: timestampLabel,
        createdAt: createdAtIso,
        ...(parsedViolations !== undefined
          ? { violations: parsedViolations }
          : {}),
      } satisfies Message;
    })
    .filter((message): message is Message => Boolean(message));
}

function formatHistoryDate(value: Date) {
  return historyDateFormatter.format(
    Number.isNaN(value.getTime()) ? new Date() : value,
  );
}

function parseChatSnapshot(
  snapshot: QueryDocumentSnapshot<DocumentData>,
): HistoryItem {
  const data = snapshot.data();
  const messages = parseMessages(data.messages);

  const titleCandidate =
    typeof data.title === "string" && data.title.trim().length > 0
      ? data.title
      : messages.find((message) => message.role === "user")?.text ?? "Conversation";

  const previewCandidate =
    typeof data.preview === "string" && data.preview.trim().length > 0
      ? data.preview
      : messages
          .slice()
          .reverse()
          .find((message) => message.role === "assistant")?.text ??
        messages[messages.length - 1]?.text ??
        "Begin a new privacy-preserving analysis.";

  let updatedAtDate = new Date();

  if (data.updatedAt && typeof data.updatedAt.toDate === "function") {
    updatedAtDate = data.updatedAt.toDate();
  } else if (messages.length > 0) {
    updatedAtDate = new Date(messages[messages.length - 1].createdAt);
  }

  return {
    id: snapshot.id,
    title: titleCandidate,
    preview: previewCandidate,
    updatedAt: formatHistoryDate(updatedAtDate),
    messages,
  };
}

export default function Home() {
  const {
    user,
    loading: authLoading,
    signOut: signOutUser,
  } = useAuth();
  const router = useRouter();

  const [messages, setMessages] = useState<Message[]>(() => createDefaultMessages());
  const [inputValue, setInputValue] = useState("");
  const [showIntro, setShowIntro] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const [chats, setChats] = useState<HistoryItem[]>([]);
  const [activeChatId, setActiveChatId] = useState<string | null>(null);
  const [pendingChatId, setPendingChatId] = useState<string | null>(null);
  const [deletingChatId, setDeletingChatId] = useState<string | null>(null);
  const [isHistoryOpen, setIsHistoryOpen] = useState(false);
  const hasInitializedActiveChat = useRef(false);

  useEffect(() => {
    if (!user) {
      setChats([]);
      setActiveChatId(null);
      setMessages(createDefaultMessages());
      setShowIntro(true);
      setPendingChatId(null);
      hasInitializedActiveChat.current = false;
      return;
    }

    const chatsQuery = query(
      collection(doc(db, "users", user.uid), "chats"),
      orderBy("updatedAt", "desc"),
    );

    const unsubscribe = onSnapshot(chatsQuery, (snapshot) => {
      const nextChats = snapshot.docs.map((document) => parseChatSnapshot(document));
      setChats(nextChats);
    });

    return () => unsubscribe();
  }, [user]);

  useEffect(() => {
    if (!user) {
      return;
    }

    if (!activeChatId) {
      if (!hasInitializedActiveChat.current && chats.length > 0) {
        const initialChat = chats[0];
        hasInitializedActiveChat.current = true;
        setActiveChatId(initialChat.id);
        setMessages(
          initialChat.messages.length > 0
            ? initialChat.messages
            : createDefaultMessages(),
        );
        setShowIntro(initialChat.messages.length === 0);
      } else if (chats.length === 0) {
        setMessages(createDefaultMessages());
        setShowIntro(true);
      }

      return;
    }

    const currentChat = chats.find((chat) => chat.id === activeChatId);

    if (!currentChat) {
      if (pendingChatId === activeChatId) {
        return;
      }

      setPendingChatId(null);
      if (chats.length > 0) {
        const fallbackChat = chats[0];
        hasInitializedActiveChat.current = true;
        setActiveChatId(fallbackChat.id);
      } else {
        setActiveChatId(null);
        setMessages(createDefaultMessages());
        setShowIntro(true);
        setPendingChatId(null);
        hasInitializedActiveChat.current = false;
      }

      return;
    }

    hasInitializedActiveChat.current = true;

    if (pendingChatId === activeChatId) {
      setPendingChatId(null);
    }

    setMessages(
      currentChat.messages.length > 0
        ? currentChat.messages
        : createDefaultMessages(),
    );
    setShowIntro(currentChat.messages.length === 0);
  }, [activeChatId, chats, pendingChatId, user]);

  const handlePersistConversation = async (
    finalMessages: Message[],
    userInput: string,
    assistantOutput: string,
    chatId: string | null,
  ) => {
    if (!user) {
      return;
    }

    try {
      const serializedMessages = finalMessages.map((message) => ({ ...message }));
      const userDocRef = doc(db, "users", user.uid);

      await setDoc(
        userDocRef,
        {
          name: user.displayName ?? "",
          email: user.email ?? "",
          updatedAt: serverTimestamp(),
        },
        { merge: true },
      );

      const previewText =
        assistantOutput.trim().slice(0, 120) || userInput.slice(0, 120);

      if (chatId) {
        const existingChat = chats.find((chat) => chat.id === chatId);
        const title = existingChat && existingChat.title.trim().length > 0
          ? existingChat.title
          : userInput.slice(0, 60) || "Conversation";

        await updateDoc(doc(userDocRef, "chats", chatId), {
          title,
          userId: user.uid,
          userName: user.displayName ?? "",
          userEmail: user.email ?? "",
          messages: serializedMessages,
          lastInput: userInput,
          lastOutput: assistantOutput,
          preview: previewText,
          updatedAt: serverTimestamp(),
        });
        setPendingChatId(null);
      } else {
        const title = userInput.slice(0, 60) || "Conversation";
        const newChatRef = await addDoc(collection(userDocRef, "chats"), {
          title,
          userId: user.uid,
          userName: user.displayName ?? "",
          userEmail: user.email ?? "",
          messages: serializedMessages,
          lastInput: userInput,
          lastOutput: assistantOutput,
          preview: previewText,
          createdAt: serverTimestamp(),
          updatedAt: serverTimestamp(),
        });

        setPendingChatId(newChatRef.id);
        setActiveChatId(newChatRef.id);
      }
    } catch (error) {
      console.error("Failed to persist conversation", error);
      setPendingChatId(null);
    }
  };

  const handleSend = async (event?: React.FormEvent<HTMLFormElement>) => {
    event?.preventDefault();

    const trimmed = inputValue.trim();

    if (!trimmed || isLoading) {
      return;
    }

    if (!user) {
      router.push("/sign-in");
      return;
    }

    const now = new Date();
    const userMessage: Message = {
      id: `user-${now.getTime()}`,
      role: "user",
      name: user.displayName ?? "You",
      text: trimmed,
      timestamp: timeFormatter.format(now),
      createdAt: now.toISOString(),
    };

    const assistantMessageId = `assistant-${now.getTime() + 1}`;
    const assistantPlaceholder: Message = {
      id: assistantMessageId,
      role: "assistant",
      name: "Nomadz AI",
      text: "Nomadz AI is preparing a privacy-screened response...",
      timestamp: timeFormatter.format(now),
      createdAt: now.toISOString(),
    };

    const optimisticMessages = [...messages, userMessage, assistantPlaceholder];

    setMessages(optimisticMessages);
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
      const assistantViolations = extractViolations(payload);
      const finalAssistantMessage: Message = {
        ...assistantPlaceholder,
        text: assistantText,
        timestamp: timeFormatter.format(new Date()),
        createdAt: new Date().toISOString(),
        ...(assistantViolations !== undefined
          ? { violations: assistantViolations }
          : {}),
      };

      const finalMessages = [
        ...optimisticMessages.slice(0, optimisticMessages.length - 1),
        finalAssistantMessage,
      ];

      setMessages(finalMessages);

      await handlePersistConversation(
        finalMessages,
        trimmed,
        assistantText,
        activeChatId,
      );
    } catch (error) {
      const fallbackError =
        error instanceof Error
          ? error.message
          : "Nomadz AI encountered an unexpected issue.";

      const failedAssistantMessage: Message = {
        ...assistantPlaceholder,
        text: `Nomadz AI ran into an issue: ${fallbackError}`,
        timestamp: timeFormatter.format(new Date()),
        createdAt: new Date().toISOString(),
      };

      const erroredMessages = [
        ...optimisticMessages.slice(0, optimisticMessages.length - 1),
        failedAssistantMessage,
      ];

      setMessages(erroredMessages);

      await handlePersistConversation(
        erroredMessages,
        trimmed,
        failedAssistantMessage.text,
        activeChatId,
      );
    } finally {
      setIsLoading(false);
    }
  };

  const handleNewChat = () => {
    if (isLoading) {
      return;
    }

    setPendingChatId(null);
    setActiveChatId(null);
    setMessages(createDefaultMessages());
    setInputValue("");
    setShowIntro(true);
  };

  const handleSelectChat = (chatId: string) => {
    if (isLoading) {
      return;
    }

    setIsHistoryOpen(false);

    if (chatId === activeChatId) {
      return;
    }

    setPendingChatId(null);
    const selectedChat = chats.find((chat) => chat.id === chatId);

    setActiveChatId(chatId);

    if (selectedChat) {
      setMessages(
        selectedChat.messages.length > 0
          ? selectedChat.messages
          : createDefaultMessages(),
      );
      setShowIntro(selectedChat.messages.length === 0);
    } else {
      setMessages(createDefaultMessages());
      setShowIntro(true);
    }
  };

  const handleDeleteChat = async (chatId: string) => {
    if (!user || deletingChatId === chatId) {
      return;
    }

    setDeletingChatId(chatId);

    try {
      await deleteDoc(doc(db, "users", user.uid, "chats", chatId));

      if (pendingChatId === chatId) {
        setPendingChatId(null);
      }

      if (activeChatId === chatId) {
        const remainingChats = chats.filter((chat) => chat.id !== chatId);

        if (remainingChats.length > 0) {
          const fallbackChat = remainingChats[0];
          hasInitializedActiveChat.current = true;
          setActiveChatId(fallbackChat.id);
          setMessages(
            fallbackChat.messages.length > 0
              ? fallbackChat.messages
              : createDefaultMessages(),
          );
          setShowIntro(fallbackChat.messages.length === 0);
        } else {
          hasInitializedActiveChat.current = false;
          setActiveChatId(null);
          setMessages(createDefaultMessages());
          setShowIntro(true);
        }
      }
    } catch (error) {
      console.error("Failed to delete chat", error);
    } finally {
      setDeletingChatId((current) => (current === chatId ? null : current));
    }
  };

  const handleSignOut = async () => {
    try {
      await signOutUser();
    } catch (error) {
      console.error("Failed to sign out", error);
    }
  };

  const isAuthenticated = useMemo(() => Boolean(user), [user]);

  const historyItems = isAuthenticated ? (
    chats.length > 0 ? (
      chats.map((chat) => {
        const isActive = chat.id === activeChatId;
        const isDeleting = deletingChatId === chat.id;

        return (
          <div key={chat.id} className="relative">
            <button
              type="button"
              onClick={() => handleSelectChat(chat.id)}
              className={`w-full rounded-2xl border px-4 py-4 pr-12 text-left transition ${
                isActive
                  ? "border-white/40 bg-white/15"
                  : "border-white/10 bg-black/30 hover:border-white/30 hover:bg-white/10"
              } disabled:cursor-not-allowed disabled:opacity-60`}
              disabled={isDeleting}
            >
              <p className="text-sm font-semibold text-white">{chat.title}</p>
              <p className="mt-1 text-xs text-white/60">{chat.preview}</p>
              <p className="mt-3 text-[11px] uppercase tracking-[0.2em] text-white/40">
                Updated {chat.updatedAt}
              </p>
            </button>
            <button
              type="button"
              onClick={(event) => {
                event.preventDefault();
                event.stopPropagation();
                void handleDeleteChat(chat.id);
              }}
              disabled={isDeleting}
              className="absolute right-4 top-4 inline-flex h-8 w-8 items-center justify-center rounded-full border border-white/10 bg-black/60 text-white/60 transition hover:border-white/30 hover:text-white focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white disabled:cursor-not-allowed disabled:opacity-60"
              aria-label="Delete chat"
            >
              <svg
                aria-hidden="true"
                className="h-4 w-4"
                viewBox="0 0 20 20"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  d="M6 6l8 8m0-8l-8 8"
                  stroke="currentColor"
                  strokeWidth="1.6"
                  strokeLinecap="round"
                />
              </svg>
              <span className="sr-only">Delete chat</span>
            </button>
          </div>
        );
      })
    ) : (
      <p className="text-sm text-white/60">Submit a query to see it saved here.</p>
    )
  ) : (
    <p className="text-sm text-white/60">
      Sign in to sync your privacy-audited chats across devices.
    </p>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-indigo-950 to-purple-900 text-slate-100">
      <div className="flex min-h-screen flex-col bg-black/20 backdrop-blur-xl">
        <header className="border-b border-white/10 bg-black/40">
          <div className="mx-auto flex w-full max-w-6xl items-center justify-between px-6 py-6">
            <div className="flex items-center gap-4">
              <button
                type="button"
                onClick={() => setIsHistoryOpen(true)}
                className="inline-flex h-11 w-11 items-center justify-center rounded-2xl border border-white/10 bg-white/10 text-white transition hover:border-white/30 hover:bg-white/20 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white lg:hidden"
                aria-label="Open chat history"
              >
                <svg
                  aria-hidden="true"
                  className="h-5 w-5"
                  viewBox="0 0 20 20"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <path
                    d="M4 5h12M4 10h12M4 15h7"
                    stroke="currentColor"
                    strokeWidth="1.6"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              </button>
              <span className="inline-flex h-12 w-12 items-center justify-center rounded-full bg-gradient-to-tr from-indigo-500 to-purple-400 text-lg font-semibold text-white shadow-lg">
                N
              </span>
              <div>
                <p className="text-lg font-semibold leading-tight">Nomadz Privacy Hub</p>
                <p className="text-xs text-white/60">
                  Answer sensitive questions with privacy-aware automation.
                </p>
              </div>
            </div>
            {authLoading ? (
              <div className="h-10 w-28 animate-pulse rounded-full bg-white/10" aria-hidden="true" />
            ) : isAuthenticated ? (
              <div className="flex items-center gap-4">
                <div className="text-right">
                  <p className="text-sm font-semibold text-white">
                    {user?.displayName ?? "Nomadz Analyst"}
                  </p>
                  <p className="text-xs text-white/60">{user?.email}</p>
                </div>
                <button
                  type="button"
                  onClick={handleSignOut}
                  className="rounded-full bg-white/10 px-5 py-2.5 text-sm font-medium text-white shadow-lg transition hover:bg-white/20 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white"
                >
                  Sign out
                </button>
              </div>
            ) : (
              <Link
                href="/sign-in"
                className="rounded-full bg-white/10 px-5 py-2.5 text-sm font-medium text-white shadow-lg transition hover:bg-white/20 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white"
              >
                Sign in
              </Link>
            )}
          </div>
        </header>

        <div className="flex flex-1 overflow-hidden">
          <aside className="hidden w-72 flex-col border-r border-white/10 bg-white/5 px-6 py-8 backdrop-blur-2xl lg:flex">
            <div className="mb-6 flex items-center justify-between">
              <h2 className="text-xs font-semibold uppercase tracking-[0.3em] text-white/70">
                Recent queries
              </h2>
              <span className="h-2 w-2 rounded-full bg-emerald-400 shadow-[0_0_12px_2px_rgba(52,211,153,0.6)]" />
            </div>
            <nav className="flex-1 space-y-3 overflow-y-auto pr-2">{historyItems}</nav>
          </aside>

          {isHistoryOpen ? (
            <div className="fixed inset-0 z-40 flex lg:hidden">
              <div
                className="absolute inset-0 bg-black/60 backdrop-blur-sm"
                role="presentation"
                onClick={() => setIsHistoryOpen(false)}
              />
              <aside className="relative ml-auto flex h-full w-80 max-w-full flex-col border-l border-white/10 bg-black/80 backdrop-blur-xl">
                <div className="flex items-center justify-between border-b border-white/10 px-6 py-4">
                  <h2 className="text-xs font-semibold uppercase tracking-[0.3em] text-white/70">
                    Recent queries
                  </h2>
                  <button
                    type="button"
                    onClick={() => setIsHistoryOpen(false)}
                    className="inline-flex h-8 w-8 items-center justify-center rounded-full border border-white/10 text-white/70 transition hover:border-white/30 hover:text-white focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white"
                    aria-label="Close chat history"
                  >
                    <svg
                      aria-hidden="true"
                      className="h-4 w-4"
                      viewBox="0 0 20 20"
                      fill="none"
                      xmlns="http://www.w3.org/2000/svg"
                    >
                      <path
                        d="M6 6l8 8m0-8l-8 8"
                        stroke="currentColor"
                        strokeWidth="1.6"
                        strokeLinecap="round"
                      />
                    </svg>
                  </button>
                </div>
                <nav className="flex-1 space-y-3 overflow-y-auto px-6 py-4">{historyItems}</nav>
              </aside>
            </div>
          ) : null}

          <main className="flex flex-1 flex-col overflow-hidden">
            <div className="mx-auto flex h-full w-full max-w-4xl flex-col gap-8 px-6 py-10">
                <section className="min-h-[64px]">
                  {showIntro ? (
                    <p className="max-w-2xl text-sm text-white/80">
                      {isAuthenticated
                        ? "Nomadz Privacy Hub coordinates policy-aware agents so you can explore sensitive datasets without exposing private details."
                        : "Sign in to Nomadz to preserve audit trails and configure privacy guardrails before asking about sensitive data."}
                    </p>
                  ) : (
                    <div className="flex items-center gap-3 rounded-3xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-white/70 shadow-lg">
                      <span className="inline-flex h-8 w-8 items-center justify-center rounded-full bg-gradient-to-tr from-indigo-500 to-sky-400 text-xs font-semibold uppercase text-white">
                        Beta
                      </span>
                      Responses are generated by the Nomadz privacy orchestration engine with layered policy checks.
                    </div>
                  )}
                </section>

                <section className="flex flex-1 flex-col overflow-hidden rounded-3xl border border-white/10 bg-black/40 shadow-2xl">
                  <div className="flex items-center justify-between border-b border-white/10 px-6 py-5">
                    <div>
                      <p className="text-xs font-semibold uppercase tracking-[0.3em] text-white/60">
                        Privacy-audited conversation
                      </p>
                      <p className="mt-1 text-sm text-white/70">
                        {isAuthenticated
                          ? "Nomadz AI applies privacy controls to every answer."
                          : "Sign in to collaborate with Nomadz AI on privacy-safe analyses."}
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
                        {message.role === "assistant" &&
                        message.violations !== undefined ? (
                          <div className="mt-3 text-xs">
                            {message.violations.length > 0 ? (
                              <>
                                <p className="font-semibold uppercase tracking-[0.2em] text-rose-200">
                                  Privacy notice
                                </p>
                                <ul className="mt-2 space-y-1 text-rose-100">
                                  {message.violations.map((violation, index) => (
                                    <li
                                      key={`${message.id}-violation-${index}`}
                                      className="flex items-start gap-2"
                                    >
                                      <span className="mt-1 h-1.5 w-1.5 flex-shrink-0 rounded-full bg-current" />
                                      <span className="flex-1">{violation}</span>
                                    </li>
                                  ))}
                                </ul>
                              </>
                            ) : (
                              <p className="flex items-center gap-2 font-medium text-emerald-300">
                                <span className="inline-flex h-2 w-2 rounded-full bg-emerald-400" />
                                Privacy check passed — no policy issues detected.
                              </p>
                            )}
                          </div>
                        ) : null}
                      </div>
                    </div>
                  ))}
                </div>

                <form onSubmit={handleSend} className="border-t border-white/10 bg-black/60 px-6 py-5">
                  <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
                    <button
                      type="button"
                      onClick={handleNewChat}
                      disabled={isLoading || !isAuthenticated}
                      className="flex h-12 w-12 shrink-0 items-center justify-center self-start rounded-2xl border border-dashed border-white/20 bg-transparent text-sm font-medium text-white/80 transition hover:border-white/50 hover:text-white focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white disabled:cursor-not-allowed disabled:opacity-60 sm:self-auto"
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
                        placeholder={
                          isAuthenticated
                            ? "Ask Nomadz AI about your sensitive data while staying compliant..."
                            : "Draft your question—sign in to run a privacy-screened analysis..."
                        }
                        className="flex-1 bg-transparent text-sm text-white placeholder:text-white/50 focus:outline-none"
                        aria-label="Type your message"
                        disabled={isLoading}
                      />
                      <button
                        type="submit"
                        disabled={isLoading || inputValue.trim().length === 0}
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
                  {!isAuthenticated ? (
                    <p className="mt-3 text-xs text-white/60">
                      Sign in to retain an auditable history of your secure conversations.
                    </p>
                  ) : null}
                </form>
              </section>
            </div>
          </main>
        </div>
      </div>
    </div>
  );
}
