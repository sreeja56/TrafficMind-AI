import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "TrafficMind AI — Intelligent Urban Traffic & Dynamic Rerouting Engine",
  description: "Next-generation traffic congestion prediction, dynamic graph rerouting, and emergency green-corridor planning engine for metropolitan road networks.",
  keywords: ["Traffic AI", "ITS", "Congestion Prediction", "Next.js", "FastAPI", "PromptWars"],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="antialiased min-h-screen flex flex-col selection:bg-cyan-500/30 selection:text-cyan-200">
        {children}
      </body>
    </html>
  );
}
