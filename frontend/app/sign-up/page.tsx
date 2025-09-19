"use client";

import Link from "next/link";
import type { FormEvent } from "react";

export default function SignUpPage() {
  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-indigo-950 to-purple-900 text-slate-100">
      <div className="flex min-h-screen flex-col items-center justify-center px-6 py-16">
        <Link
          href="/"
          className="mb-10 inline-flex items-center gap-3 text-sm font-medium text-white/70 transition hover:text-white"
        >
          <svg
            aria-hidden="true"
            className="h-4 w-4"
            viewBox="0 0 20 20"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path
              d="m12.5 5-5 5 5 5"
              stroke="currentColor"
              strokeWidth="1.6"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
          Back to chats
        </Link>

        <div className="w-full max-w-md rounded-3xl border border-white/10 bg-black/60 p-10 shadow-2xl backdrop-blur-xl">
          <div className="mb-8 text-center">
            <h1 className="text-2xl font-semibold">Create your Nomadz account</h1>
            <p className="mt-3 text-sm text-white/70">
              Sign up to sync travel chats, itineraries, and insights across every device.
            </p>
          </div>

          <form onSubmit={handleSubmit} className="space-y-5">
            <div className="space-y-2">
              <label
                className="text-xs font-semibold uppercase tracking-[0.3em] text-white/60"
                htmlFor="name"
              >
                Full name
              </label>
              <input
                id="name"
                name="name"
                type="text"
                required
                autoComplete="name"
                className="w-full rounded-2xl border border-white/10 bg-white/10 px-4 py-3 text-sm text-white placeholder:text-white/50 focus:border-indigo-400 focus:outline-none focus:ring-2 focus:ring-indigo-500/50"
                placeholder="Your name"
              />
            </div>

            <div className="space-y-2">
              <label
                className="text-xs font-semibold uppercase tracking-[0.3em] text-white/60"
                htmlFor="email"
              >
                Email
              </label>
              <input
                id="email"
                name="email"
                type="email"
                required
                autoComplete="email"
                className="w-full rounded-2xl border border-white/10 bg-white/10 px-4 py-3 text-sm text-white placeholder:text-white/50 focus:border-indigo-400 focus:outline-none focus:ring-2 focus:ring-indigo-500/50"
                placeholder="you@example.com"
              />
            </div>

            <div className="space-y-2">
              <label
                className="text-xs font-semibold uppercase tracking-[0.3em] text-white/60"
                htmlFor="password"
              >
                Password
              </label>
              <input
                id="password"
                name="password"
                type="password"
                required
                autoComplete="new-password"
                className="w-full rounded-2xl border border-white/10 bg-white/10 px-4 py-3 text-sm text-white placeholder:text-white/50 focus:border-indigo-400 focus:outline-none focus:ring-2 focus:ring-indigo-500/50"
                placeholder="Create a secure password"
              />
            </div>

            <div className="space-y-2">
              <label
                className="text-xs font-semibold uppercase tracking-[0.3em] text-white/60"
                htmlFor="confirm"
              >
                Confirm password
              </label>
              <input
                id="confirm"
                name="confirm"
                type="password"
                required
                autoComplete="new-password"
                className="w-full rounded-2xl border border-white/10 bg-white/10 px-4 py-3 text-sm text-white placeholder:text-white/50 focus:border-indigo-400 focus:outline-none focus:ring-2 focus:ring-indigo-500/50"
                placeholder="Repeat your password"
              />
            </div>

            <button
              type="submit"
              className="w-full rounded-2xl bg-gradient-to-r from-indigo-500 via-purple-500 to-sky-500 px-4 py-3 text-sm font-semibold text-white shadow-lg transition hover:from-indigo-400 hover:via-purple-400 hover:to-sky-400 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white"
            >
              Create account
            </button>
          </form>

          <p className="mt-8 text-center text-sm text-white/70">
            Already have an account? 
            <Link href="/sign-in" className="font-semibold text-indigo-200 transition hover:text-indigo-100">
              Sign in
            </Link>
          </p>
        </div>
      </div>
    </div>
  );
}
