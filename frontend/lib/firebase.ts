"use client";

import { getApps, initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";
import { getFirestore } from "firebase/firestore";

type FirebaseConfig = {
  apiKey: string;
  authDomain: string;
  projectId: string;
  storageBucket: string;
  messagingSenderId: string;
  appId: string;
};

function getFirebaseConfig(): FirebaseConfig {
  const config: Partial<FirebaseConfig> = {
    apiKey: process.env.NEXT_PUBLIC_FIREBASE_API_KEY,
    authDomain: process.env.NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN,
    projectId: process.env.NEXT_PUBLIC_FIREBASE_PROJECT_ID,
    storageBucket: process.env.NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET,
    messagingSenderId: process.env.NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID,
    appId: process.env.NEXT_PUBLIC_FIREBASE_APP_ID,
  };

  const missingKeys = Object.entries(config)
    .filter(([, value]) => !value)
    .map(([key]) => key);

  if (missingKeys.length > 0) {
    throw new Error(
      `Missing Firebase configuration values: ${missingKeys.join(", ")}. ` +
        "Ensure the NEXT_PUBLIC_FIREBASE_* environment variables are set.",
    );
  }

  return config as FirebaseConfig;
}

const firebaseApp = getApps().length > 0
  ? getApps()[0]
  : initializeApp(getFirebaseConfig());

export const auth = getAuth(firebaseApp);
export const db = getFirestore(firebaseApp);

