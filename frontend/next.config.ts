import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // API proxy: forward /api/* to FastAPI backend
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000"}/api/:path*`,
      },
    ];
  },
  // Image optimization
  images: {
    formats: ["image/avif", "image/webp"],
  },
  // Transpile plotly.js for SSR compatibility
  transpilePackages: ["react-plotly.js", "plotly.js"],
};

export default nextConfig;
