import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "#080d1a",
        foreground: "#f8fafc",
        card: "rgba(15, 23, 42, 0.75)",
        cyber: {
          cyan: "#00E5FF",
          green: "#00E676",
          amber: "#FFAB00",
          red: "#FF1744",
          purple: "#7C4DFF",
        },
      },
    },
  },
  plugins: [],
};

export default config;
