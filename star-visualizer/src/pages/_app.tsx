import { VisualizerProvider } from '@/context/VisualizerContext'
import '@/styles/globals.css'  // If you have global styles
import type { AppProps } from 'next/app'

function MyApp({ Component, pageProps }: AppProps) {
  return (
    <VisualizerProvider>
      <Component {...pageProps} />
    </VisualizerProvider>
  )
}

export default MyApp
