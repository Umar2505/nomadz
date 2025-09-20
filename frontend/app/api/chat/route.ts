import { NextResponse } from "next/server";

const API_BASE_URL =
  process.env.NOMADZ_API_BASE_URL ||
  process.env.API_BASE_URL ||
  process.env.NEXT_PUBLIC_API_BASE_URL ||
  "http://127.0.0.1:5000";

const API_ENDPOINT = `${API_BASE_URL.replace(/\/$/, "")}/api`;

export async function POST(request: Request) {
  let payload: unknown = {};

  try {
    console.log("Parsing request JSON payload");
    payload = await request.json();
  } catch {
    // Ignore JSON parsing errors and fall back to an empty payload
  }

  try {
    const upstreamResponse = await fetch(API_ENDPOINT, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload ?? {}),
    });
    console.log("Received response from Nomadz API");

    let data: unknown;

    try {
      data = await upstreamResponse.json();
    } catch {
      data = undefined;
    }

    if (!upstreamResponse.ok) {
      const errorMessage =
        typeof data === "object" && data && "error" in data &&
          typeof (data as { error?: unknown }).error === "string"
          ? (data as { error: string }).error
          : "The Nomadz API returned an unexpected error.";

      return NextResponse.json(
        { error: errorMessage },
        {
          status: upstreamResponse.status,
        },
      );
    }

    return NextResponse.json(
      data ?? {},
      {
        status: upstreamResponse.status,
      },
    );
  } catch (error) {
    console.error("Failed to reach Nomadz API", error);

    return NextResponse.json(
      { error: "Unable to reach the Nomadz API. Please try again in a moment." },
      { status: 502 },
    );
  }
}
