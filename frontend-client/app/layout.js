import { Inter } from 'next/font/google'
import './ui/globals.css'
import { ThemeProvider } from './contexts/ThemeContext';
import { NotificationProvider } from './contexts/NotificationContext';

const inter = Inter({ subsets: ['latin'] })

export const metadata = {
  title: 'MDS25 Dashboard',
  description: 'Hybrid Edge-Cloud Architecture for AI Model Deployment',
}

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>
        <ThemeProvider>
          <NotificationProvider>
            {children}
          </NotificationProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}