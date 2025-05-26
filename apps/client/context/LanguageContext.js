import React, { createContext, useState, useContext, useEffect } from 'react';
import AsyncStorage from '@react-native-async-storage/async-storage';

const LanguageContext = createContext();

export const LanguageProvider = ({ children }) => {
  const [idioma, setIdioma] = useState('es');

  useEffect(() => {
    AsyncStorage.getItem('idioma').then(lang => {
      if (lang) setIdioma(lang);
    });
  }, []);

  useEffect(() => {
    AsyncStorage.setItem('idioma', idioma);
  }, [idioma]);

  return (
    <LanguageContext.Provider value={{ idioma, setIdioma }}>
      {children}
    </LanguageContext.Provider>
  );
};

export const useLanguage = () => useContext(LanguageContext);