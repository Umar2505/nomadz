"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState, type FormEvent } from "react";
import { FirebaseError } from "firebase/app";
import { signInWithEmailAndPassword } from "firebase/auth";

import { auth } from "@/lib/firebase";

function getErrorMessage(error: unknown) {
  if (error instanceof FirebaseError) {
    switch (error.code) {
      case "auth/invalid-email":
        return "The email address appears to be invalid.";
      case "auth/user-disabled":
        return "This account has been disabled. Please contact support.";
      case "auth/user-not-found":
      case "auth/wrong-password":
        return "Incorrect email or password.";
      case "auth/too-many-requests":
        return "Too many failed attempts. Please try again later.";
      default:
        return "Unable to sign in with the provided credentials.";
    }
  }

  return "We couldn\u2019t sign you in. Please try again.";
}

export default function SignInPage() {
  const router = useRouter();
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    const formData = new FormData(event.currentTarget);
    const email = String(formData.get("email") ?? "").trim();
    const password = String(formData.get("password") ?? "");

    if (!email || !password) {
      setErrorMessage("Please provide both your email and password.");
      return;
    }

    try {
      setIsSubmitting(true);
      setErrorMessage(null);

      await signInWithEmailAndPassword(auth, email, password);

      router.push("/");
    } catch (error) {
      setErrorMessage(getErrorMessage(error));
    } finally {
      setIsSubmitting(false);
    }
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
            <h1 className="text-2xl font-semibold">Sign in to Nomadz</h1>
            <p className="mt-3 text-sm text-white/70">
              Access your saved itineraries and pick up conversations without missing a beat.
            </p>
          </div>

          <form onSubmit={handleSubmit} className="space-y-5" noValidate>
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
                aria-invalid={errorMessage ? "true" : "false"}
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
                autoComplete="current-password"
                className="w-full rounded-2xl border border-white/10 bg-white/10 px-4 py-3 text-sm text-white placeholder:text-white/50 focus:border-indigo-400 focus:outline-none focus:ring-2 focus:ring-indigo-500/50"
                placeholder="Enter your password"
                aria-invalid={errorMessage ? "true" : "false"}
              />
            </div>

            {errorMessage ? (
              <p className="text-sm text-rose-300" role="alert" aria-live="polite">
                {errorMessage}
              </p>
            ) : null}

            <button
              type="submit"
              disabled={isSubmitting}
              className="w-full rounded-2xl bg-gradient-to-r from-indigo-500 via-purple-500 to-sky-500 px-4 py-3 text-sm font-semibold text-white shadow-lg transition hover:from-indigo-400 hover:via-purple-400 hover:to-sky-400 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white disabled:cursor-not-allowed disabled:opacity-60"
            >
              {isSubmitting ? "Signing in..." : "Sign in"}
            </button>
          </form>

          <p className="mt-8 text-center text-sm text-white/70">
            New to Nomadz?{" "}
            <Link href="/sign-up" className="font-semibold text-indigo-200 transition hover:text-indigo-100">
              Create an account
            </Link>
          </p>
        </div>
      </div>
    </div>
  );
}
