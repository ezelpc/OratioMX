import React, { useContext, useState, useCallback } from 'react';
import {
  View,
  TouchableOpacity,
  Image,
  Text,
  ScrollView,
  Dimensions,
} from 'react-native';
import { useFocusEffect } from '@react-navigation/native';
import GlobalStyles from '../static/global';
import Layout from '../layout/Layout';
import Label from '../components/CustomLabel';
import Ionicons from 'react-native-vector-icons/Ionicons';
import MaterialIcons from 'react-native-vector-icons/MaterialIcons';
import { AuthContext } from '../context/AuthContext';
import Colors from '../static/colors';
import { useLanguage } from '../context/LanguageContext';
const { width } = Dimensions.get('window');

const textos = {
  es: {
    traducciones: 'Traducciones realizadas',
    videos: 'Videos subidos',
    comunidad: 'Comunidad',
    reconocimientos: 'Reconocimientos',
    traducir: 'Ir a Traducir',
    bienvenida: 'Bienvenido a OratioMX, tu traductor de lenguaje de señas. Consulta tus métricas, sube videos y participa en la comunidad.',
    editarPerfil: 'Editar perfil',
    configuracion: 'Configuración',
    comentarios: 'Comentarios',
    ayuda: 'Ayuda',
    cerrarSesion: 'Cerrar sesión',
    saludo: (hora) => {
      if (hora >= 5 && hora < 12) return 'Buenos días';
      if (hora >= 12 && hora < 20) return 'Buenas tardes';
      return 'Buenas noches';
    },
  },
  en: {
    traducciones: 'Translations made',
    videos: 'Videos uploaded',
    comunidad: 'Community',
    reconocimientos: 'Achievements',
    traducir: 'Go to Translate',
    bienvenida: 'Welcome to OratioMX, your sign language translator. Check your metrics, upload videos, and join the community.',
    editarPerfil: 'Edit profile',
    configuracion: 'Settings',
    comentarios: 'Feedback',
    ayuda: 'Help',
    cerrarSesion: 'Log out',
    saludo: (hora) => {
      if (hora >= 5 && hora < 12) return 'Good morning';
      if (hora >= 12 && hora < 20) return 'Good afternoon';
      return 'Good evening';
    },
  }
};

const Welcome = ({ navigation }) => {
  const { user, logout } = useContext(AuthContext);
  const [menuVisible, setMenuVisible] = useState(false);
  const { idioma, setIdioma } = useLanguage();
  useFocusEffect(
    useCallback(() => {
      setMenuVisible(false);
    }, [])
  );

  let nombre = user?.nombres || user?.usuario || 'Usuario';
  if (nombre.startsWith('Usuario:')) {
    nombre = nombre.replace('Usuario:', '').trim();
  }

  let foto = user?.foto;
  if (foto) {
    if (!foto.startsWith('http')) {
      foto = 'http://TU_DOMINIO_O_IP:PUERTO' + foto;
    }
  } else {
    foto = 'https://ui-avatars.com/api/?name=' + encodeURIComponent(nombre);
  }

  const hora = new Date().getHours();
  let saludo = textos[idioma]?.saludo(hora) || '¡Hola';

  const handleLogout = () => {
    setMenuVisible(false);
    logout();
  };

  const handleEditProfile = () => {
    setMenuVisible(false);
    navigation.navigate('EditProfile', { ...user });
  };

  const metrics = [
    { icon: 'hand-left-outline', label: textos[idioma]?.traducciones || 'Traducciones realizadas', value: '128' },
    { icon: 'videocam-outline', label: textos[idioma]?.videos || 'Videos subidos', value: '34' },
    { icon: 'people-outline', label: textos[idioma]?.comunidad || 'Comunidad', value: '1,250' },
    { icon: 'star-outline', label: textos[idioma]?.reconocimientos || 'Reconocimientos', value: '8' },
  ];

  return (
    <Layout>
      <ScrollView contentContainerStyle={[GlobalStyles.container, { backgroundColor: Colors.bg || '#f5f6fa' }]}>
        {/* Header */}
        <View style={[GlobalStyles.headerRow, { marginBottom: 10 }]}>
          <Label text="Oratio Mx" style={[GlobalStyles.title, { color: Colors.primary }]} />
          <TouchableOpacity onPress={() => setMenuVisible(!menuVisible)}>
            <Ionicons name="menu" size={32} color={Colors.primary} />
          </TouchableOpacity>
        </View>

        {/* Menú lateral tipo Messenger */}
        {menuVisible && (
          <View style={[
            GlobalStyles.menuContainer,
            {
              width: width * 0.75,
              backgroundColor: '#fff',
              borderRadius: 24,
              shadowColor: Colors.primary,
              shadowOpacity: 0.08,
              shadowRadius: 10,
              elevation: 8,
            }
          ]}>
            <View style={[GlobalStyles.menuHeader, { alignItems: 'center' }]}>
              <Image source={{ uri: foto }} style={[GlobalStyles.avatar, { borderColor: Colors.primary }]} />
              <Label text={`${saludo},`} style={[GlobalStyles.menuSaludo, { color: Colors.primary }]} />
              <Label text={nombre} style={[GlobalStyles.menuNombre, { color: Colors.primary }]} />
            </View>
            <TouchableOpacity style={GlobalStyles.menuOption} onPress={handleEditProfile}>
              <Ionicons name="person-circle-outline" size={22} color={Colors.primary} />
              <Text style={GlobalStyles.menuText}>{textos[idioma]?.editarPerfil || 'Editar perfil'}</Text>
            </TouchableOpacity>
            <TouchableOpacity style={GlobalStyles.menuOption} onPress={() => navigation.navigate('Settings')}>
              <Ionicons name="settings-outline" size={22} color={Colors.primary} />
              <Text style={GlobalStyles.menuText}>{textos[idioma]?.configuracion || 'Configuración'}</Text>
            </TouchableOpacity>
            <TouchableOpacity style={GlobalStyles.menuOption} onPress={() => navigation.navigate('Coments')}>
              <MaterialIcons name="feedback" size={22} color={Colors.primary} />
              <Text style={GlobalStyles.menuText}>{textos[idioma]?.comentarios || 'Comentarios'}</Text>
            </TouchableOpacity>
            <TouchableOpacity style={GlobalStyles.menuOption} onPress={() => navigation.navigate('Help')}>
              <Ionicons name="help-circle-outline" size={22} color={Colors.primary} />
              <Text style={GlobalStyles.menuText}>{textos[idioma]?.ayuda || 'Ayuda'}</Text>
            </TouchableOpacity>
            <TouchableOpacity style={GlobalStyles.menuOption} onPress={handleLogout}>
              <MaterialIcons name="logout" size={22} color="#ff4d4d" />
              <Text style={[GlobalStyles.menuText, { color: '#ff4d4d' }]}>{textos[idioma]?.cerrarSesion || 'Cerrar sesión'}</Text>
            </TouchableOpacity>
          </View>
        )}

        {/* Métricas tipo Messenger */}
        <View style={GlobalStyles.metricsRow}>
          {metrics.map((m, i) => (
            <View
              key={i}
              style={[
                GlobalStyles.metricCard,
                {
                  backgroundColor: '#fff',
                  borderRadius: 18,
                  borderWidth: 1,
                  borderColor: '#e4e6eb',
                  shadowColor: Colors.primary,
                  shadowOpacity: 0.06,
                  shadowRadius: 4,
                  elevation: 2,
                }
              ]}
            >
              <Ionicons name={m.icon} size={32} color={Colors.primary} style={{ marginBottom: 6 }} />
              <Text style={[GlobalStyles.metricValue, { color: Colors.primary }]}>{m.value}</Text>
              <Text style={[GlobalStyles.metricLabel, { color: Colors.text || '#222' }]}>{m.label}</Text>
            </View>
          ))}
        </View>

        {/* Botón traducir tipo Messenger */}
        <TouchableOpacity
          style={[
            GlobalStyles.translateButton,
            {
              backgroundColor: Colors.primary,
              borderRadius: 18,
              marginTop: 10,
              marginBottom: 18,
            }
          ]}
          onPress={() => navigation.navigate('Translate')}
        >
          <Ionicons name="language-outline" size={24} color="#fff" />
          <Text style={[GlobalStyles.translateButtonText, { color: '#fff' }]}>{textos[idioma]?.traducir || 'Ir a Traducir'}</Text>
        </TouchableOpacity>

        {/* Mensaje bienvenida */}
        <View>
          <Label
            text={textos[idioma]?.bienvenida || 'Bienvenido a OratioMX, tu traductor de lenguaje de señas. Consulta tus métricas, sube videos y participa en la comunidad.'}
            style={[GlobalStyles.welcomeMsg, { color: Colors.text || '#222', backgroundColor: 'transparent' }]}
          />
        </View>
      </ScrollView>
    </Layout>
  );
};

export default Welcome;
