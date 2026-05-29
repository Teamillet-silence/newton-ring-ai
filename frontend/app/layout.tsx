import "./globals.css";

export const metadata = {
  title: "牛顿环 AI 助手",
  description: "Newton Ring AI",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {

  return (
    <html lang="zh-CN">
      <body>
        {children}
      </body>
    </html>
  );
}