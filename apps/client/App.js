import React from 'react';
import { LanguageProvider } from './context/LanguageContext';
import AppNavigator from './AppNavigator';
import { AuthProvider } from './context/AuthContext';
import { ThemeProvider } from './context/ThemeContext';

export default function App() {
  return (
    <LanguageProvider>
      <AuthProvider>
        <ThemeProvider>
          <AppNavigator />
        </ThemeProvider>
      </AuthProvider>
    </LanguageProvider>
  );
}
