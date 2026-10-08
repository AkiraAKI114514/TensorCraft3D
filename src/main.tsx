import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import ErrorBoundary from './ErrorBoundary';
import { I18nProvider } from './i18n';

ReactDOM.createRoot(document.getElementById('root')!).render(<React.StrictMode><I18nProvider><ErrorBoundary><App /></ErrorBoundary></I18nProvider></React.StrictMode>);
