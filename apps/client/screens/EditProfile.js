import React, { useContext, useState } from 'react';
import { View, Text, TextInput, TouchableOpacity, ActivityIndicator, Alert, Image } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import Ionicons from 'react-native-vector-icons/Ionicons';
import Layout from '../layout/Layout';
import Label from '../components/CustomLabel';
import { AuthContext } from '../context/AuthContext';
import GlobalStyles from '../static/global';
import Colors from '../static/colors';
import { useLanguage } from '../context/LanguageContext';

const textos = {
  es: {
    editarPerfil: 'Editar Perfil',
    fotoPerfil: 'Foto de perfil',
    cambiarFoto: 'Cambiar foto',
    agregarFoto: 'Agregar foto',
    usuario: 'Usuario',
    placeholderUsuario: 'Usuario',
    contraseña: 'Contraseña',
    placeholderContraseña: 'Nueva contraseña',
    guardar: 'Guardar cambios',
    exito: 'Perfil actualizado correctamente',
    error: 'No se pudo actualizar el perfil',
    errorConexion: 'No se pudo conectar al servidor',
    permiso: 'Permiso requerido',
    permisoMsg: 'Se requiere permiso para acceder a tus fotos.',
  },
  en: {
    editarPerfil: 'Edit Profile',
    fotoPerfil: 'Profile photo',
    cambiarFoto: 'Change photo',
    agregarFoto: 'Add photo',
    usuario: 'Username',
    placeholderUsuario: 'Username',
    contraseña: 'Password',
    placeholderContraseña: 'New password',
    guardar: 'Save changes',
    exito: 'Profile updated successfully',
    error: 'Could not update profile',
    errorConexion: 'Could not connect to server',
    permiso: 'Permission required',
    permisoMsg: 'Permission to access your photos is required.',
  }
};

const EditProfile = () => {
  const { user } = useContext(AuthContext);
  const { idioma } = useLanguage();

  const [foto, setFoto] = useState(user?.foto || '');
  const [usuario, setUsuario] = useState(user?.usuario || '');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);

  const API_URL = 'http://192.168.1.69:3000/api/usuarios/' + user?.id;

  const handlePickImage = async () => {
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (status !== 'granted') {
      Alert.alert(textos[idioma].permiso, textos[idioma].permisoMsg);
      return;
    }
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [1, 1],
      quality: 0.7,
      base64: false,
    });
    if (!result.canceled && result.assets && result.assets.length > 0) {
      setFoto(result.assets[0].uri);
    }
  };

  const handleGuardar = async () => {
    setLoading(true);
    try {
      const res = await fetch(API_URL, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          foto,
          usuario,
          password: password || undefined,
        }),
      });
      const data = await res.json();
      if (res.ok) {
        Alert.alert(textos[idioma].exito);
        // Aquí podrías actualizar el contexto o navegar
      } else {
        Alert.alert('Error', data.error || textos[idioma].error);
      }
    } catch (error) {
      Alert.alert('Error', textos[idioma].errorConexion);
    }
    setLoading(false);
  };

  return (
    <Layout>
      <View style={[GlobalStyles.container, { justifyContent: 'center', paddingTop: 30 }]}>
        <Label text={textos[idioma].editarPerfil} style={[GlobalStyles.title, { color: Colors.primary }]} />

        <Text style={GlobalStyles.label}>{textos[idioma].fotoPerfil}</Text>
        <TouchableOpacity onPress={handlePickImage} style={{ alignItems: 'center', marginBottom: 16 }}>
          {foto ? (
            <Image
              source={{ uri: foto }}
              style={[
                GlobalStyles.avatar,
                { borderColor: Colors.primary, borderWidth: 2 }
              ]}
            />
          ) : (
            <View style={[
              GlobalStyles.avatarPlaceholder,
              { borderColor: Colors.primary, backgroundColor: Colors.bgSecondary || '#e4e6eb' }
            ]}>
              <Ionicons name="camera-outline" size={32} color={Colors.primary} />
            </View>
          )}
          <Text style={{ color: Colors.primary, marginTop: 4 }}>
            {foto ? textos[idioma].cambiarFoto : textos[idioma].agregarFoto}
          </Text>
        </TouchableOpacity>

        <Text style={GlobalStyles.label}>{textos[idioma].usuario}</Text>
        <TextInput
          style={{
            ...GlobalStyles.input,
            backgroundColor: '#fff',
            borderRadius: 24,
            borderWidth: 1,
            borderColor: Colors.primary,
            color: Colors.text,
            fontSize: 16,
            marginBottom: 12,
          }}
          value={usuario}
          onChangeText={setUsuario}
          placeholder={textos[idioma].placeholderUsuario}
          autoCapitalize="none"
        />

        <Text style={GlobalStyles.label}>{textos[idioma].contraseña}</Text>
        <TextInput
          style={{
            ...GlobalStyles.input,
            backgroundColor: '#fff',
            borderRadius: 24,
            borderWidth: 1,
            borderColor: Colors.primary,
            color: Colors.text,
            fontSize: 16,
            marginBottom: 18,
          }}
          value={password}
          onChangeText={setPassword}
          placeholder={textos[idioma].placeholderContraseña}
          secureTextEntry
          autoCapitalize="none"
        />

        <TouchableOpacity
          style={[
            GlobalStyles.button,
            {
              backgroundColor: Colors.primary,
              borderRadius: 24,
              marginTop: 10,
              marginBottom: 10,
            }
          ]}
          onPress={handleGuardar}
          disabled={loading}
        >
          {loading ? (
            <ActivityIndicator color="#fff" />
          ) : (
            <Text style={GlobalStyles.buttonText}>{textos[idioma].guardar}</Text>
          )}
        </TouchableOpacity>
      </View>
    </Layout>
  );
};

export default EditProfile;