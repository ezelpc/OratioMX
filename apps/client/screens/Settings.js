import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, Switch, TouchableOpacity } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import Layout from '../layout/Layout';
import Label from '../components/CustomLabel';
import GlobalStyles from '../static/global';
import { useLanguage } from '../context/LanguageContext';
import { useTheme } from '../context/ThemeContext';

const Settings = () => {
  const [notificaciones, setNotificaciones] = useState(true);
  const { idioma, setIdioma } = useLanguage(); // Usa el contexto
  const { darkMode, setDarkMode } = useTheme();

  // Cargar preferencias al iniciar
  useEffect(() => {
    const cargarPreferencias = async () => {
      const notif = await AsyncStorage.getItem('notificaciones');
      const tema = await AsyncStorage.getItem('temaOscuro');
      if (notif !== null) setNotificaciones(JSON.parse(notif));
      if (tema !== null) setDarkMode(JSON.parse(tema));
    };
    cargarPreferencias();
  }, []);

  // Guardar preferencias cuando cambian
  useEffect(() => {
    AsyncStorage.setItem('notificaciones', JSON.stringify(notificaciones));
    AsyncStorage.setItem('temaOscuro', JSON.stringify(darkMode));
  }, [notificaciones, darkMode]);

  const textos = {
    es: {
      config: 'Configuración',
      notificaciones: 'Notificaciones',
      tema: 'Tema oscuro',
      idioma: 'Idioma',
      espanol: 'Español',
      ingles: 'English',
    },
    en: {
      config: 'Settings',
      notificaciones: 'Notifications',
      tema: 'Dark theme',
      idioma: 'Language',
      espanol: 'Spanish',
      ingles: 'English',
    },
  };

  return (
    <Layout darkMode={darkMode}>
      <View style={GlobalStyles.container}>
        <Label text={textos[idioma].config} style={GlobalStyles.title} />

        <View style={GlobalStyles.optionRow}>
          <Text style={GlobalStyles.optionText}>{textos[idioma].notificaciones}</Text>
          <Switch
            value={notificaciones}
            onValueChange={setNotificaciones}
            thumbColor={notificaciones ? '#00ffe7' : '#ccc'}
            trackColor={{ false: '#555', true: '#00ffe7' }}
          />
        </View>

        <View style={GlobalStyles.optionRow}>
          <Text style={GlobalStyles.optionText}>{textos[idioma].tema}</Text>
          <Switch
            value={darkMode}
            onValueChange={setDarkMode}
            thumbColor={darkMode ? '#00ffe7' : '#ccc'}
            trackColor={{ false: '#555', true: '#00ffe7' }}
          />
        </View>

        <View style={GlobalStyles.optionRow}>
          <Text style={GlobalStyles.optionText}>{textos[idioma].idioma}</Text>
          <TouchableOpacity
            style={GlobalStyles.languageButton}
            onPress={() => setIdioma(idioma === 'es' ? 'en' : 'es')}
          >
            <Text style={GlobalStyles.languageText}>
              {idioma === 'es' ? textos[idioma].espanol : textos[idioma].ingles}
            </Text>
          </TouchableOpacity>
        </View>
      </View>
    </Layout>
  );
};

export default Settings;