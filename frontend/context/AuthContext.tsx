"use client";

import {
  type User,
  onAuthStateChanged,
  signOut as firebaseSignOut,
} from "firebase/auth";
import { doc, serverTimestamp, setDoc } from "firebase/firestore";
import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";

import { auth, db } from "@/lib/firebase";

type AuthContextValue = {
  user: User | null;
  loading: boolean;
  signOut: () => Promise<void>;
};

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, async (firebaseUser) => {
      setUser(firebaseUser);

      if (firebaseUser) {
        const profile = {
          name: firebaseUser.displayName ?? "",
          email: firebaseUser.email ?? "",
          updatedAt: serverTimestamp(),
        };

        try {
          await setDoc(doc(db, "users", firebaseUser.uid), profile, {
            merge: true,
          });
        } catch (error) {
          console.error("Failed to sync user profile in Firestore", error);
        }
      }

      setLoading(false);
    });

    return () => unsubscribe();
  }, []);

  const value = useMemo<AuthContextValue>(
    () => ({
      user,
      loading,
      signOut: async () => {
        await firebaseSignOut(auth);
      },
    }),
    [loading, user],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);

  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider");
  }

  return context;
}

