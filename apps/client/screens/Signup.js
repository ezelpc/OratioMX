import React, { useState } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  ScrollView,
  KeyboardAvoidingView,
  Platform,
  Alert,
  ActivityIndicator,
  Image,
} from 'react-native';
import DateTimePicker from '@react-native-community/datetimepicker';
import Ionicons from 'react-native-vector-icons/Ionicons';
import * as ImagePicker from 'expo-image-picker';
  import Layout from '../layout/Layout';
import CustomInput from '../components/CustomInput';
import PrimaryButton from '../components/CustomButton';
import Label from '../components/CustomLabel';
import LinkText from '../components/CustomLink';
import GlobalStyles from '../static/global';
import Colors from '../static/colors';

const Signup = ({ navigation }) => {
  const [name, setName] = useState('');
  const [papellido, setPapellido] = useState('');
  const [sapellido, setSapellido] = useState('');
  const [fechaNacimiento, setFechaNacimiento] = useState('');
  const [usuario, setUsuario] = useState('');
  const [correo, setCorreo] = useState('');
  const [contraseña, setContraseña] = useState('');
  const [confirmarContraseña, setConfirmarContraseña] = useState('');
  const [mostrarContraseña, setMostrarContraseña] = useState(false);
  const [mostrarConfirmarContraseña, setMostrarConfirmarContraseña] = useState(false);
  const [aceptaTerminos, setAceptaTerminos] = useState(false);
  const [mostrarPicker, setMostrarPicker] = useState(false);
  const [fechaDate, setFechaDate] = useState(new Date());
  const [loading, setLoading] = useState(false);
  const [foto, setFoto] = useState(null);

  const handleOpenTerminos = () => {
    navigation.navigate('Terminos', {
      onAccept: (aceptado) => setAceptaTerminos(aceptado),
    });
  };

  const handleDateChange = (event, selectedDate) => {
    setMostrarPicker(false);
    if (selectedDate) {
      const formatted = selectedDate.toISOString().split('T')[0];
      setFechaNacimiento(formatted);
      setFechaDate(selectedDate);
    }
  };

  const handlePickImage = async () => {
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

  const handleSignup = async () => {
    const isEmpty = (str) => !str || str.trim().length === 0;

    if (
      isEmpty(name) ||
      isEmpty(papellido) ||
      isEmpty(usuario) ||
      isEmpty(correo) ||
      isEmpty(contraseña) ||
      isEmpty(confirmarContraseña)
    ) {
      return Alert.alert('Error', 'Por favor completa todos los campos obligatorios.');
    }

    if (contraseña !== confirmarContraseña) {
      return Alert.alert('Error', 'Las contraseñas no coinciden.');
    }

    if (!aceptaTerminos) {
      return Alert.alert('Aviso', 'Debes aceptar los términos y condiciones para continuar.');
    }

    setLoading(true);

    try {
      const response = await fetch('http://192.168.1.69:3000/api/auth/register',
      //const response = await fetch('http://192.168.1.81:3000/api/auth/register',  
        {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          nombres: name.trim(),
          primer_apellido: papellido.trim(),
          segundo_apellido: sapellido.trim(),
          fecha_nacimiento: fechaNacimiento,
          usuario: usuario.trim(),
          correo_electronico: correo.trim(),
          contrasena: contraseña,
          rol: 'Usuario',
          foto: foto, // Se envía la URI de la foto
        }),
      });

      const data = await response.json();

      if (response.ok || response.status === 201) {
        Alert.alert('Éxito', 'Usuario registrado correctamente.', [
          {
            text: 'OK',
            onPress: () =>
              navigation.reset({
                index: 0,
                routes: [{ name: 'Welcome' }],
              }),
          },
        ]);
      } else {
        Alert.alert('Error', data.error || 'Ocurrió un error inesperado al registrar.');
      }
    } catch (error) {
      console.error('Error al registrar:', error);
      Alert.alert('Error', 'No se pudo conectar al servidor. Intenta más tarde.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Layout>
      <KeyboardAvoidingView
        style={{ flex: 1 }}
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
      >
        <ScrollView
          contentContainerStyle={{ padding: 20, paddingBottom: 120 }}
          keyboardShouldPersistTaps="handled"
          showsVerticalScrollIndicator={false}
        >
          <Label text="Nombre(s)" />
          <CustomInput placeholder="Nombre" value={name} onChangeText={setName} />

          <Label text="Primer Apellido" />
          <CustomInput placeholder="Apellido paterno" value={papellido} onChangeText={setPapellido} />

          <Label text="Segundo Apellido" />
          <CustomInput placeholder="Apellido materno" value={sapellido} onChangeText={setSapellido} />

          <Label text="Foto de Perfil" />
          <TouchableOpacity onPress={handlePickImage} style={{ alignItems: 'center', marginBottom: 16 }}>
            {foto ? (
              <Image
                source={{ uri: foto }}
                style={{
                  width: 80,
                  height: 80,
                  borderRadius: 40,
                  marginBottom: 8,
                  borderWidth: 2,
                  borderColor: Colors.primary,
                }}
              />
            ) : (
              <View
                style={{
                  width: 80,
                  height: 80,
                  borderRadius: 40,
                  backgroundColor: Colors.bgSecondary || '',
                  justifyContent: 'center',
                  alignItems: 'center',
                  marginBottom: 8,
                  borderWidth: 2,
                  borderColor: Colors.primary,
                }}
              >
                <Ionicons name="camera-outline" size={32} color={Colors.primary} />
              </View>
            )}
            <Text style={{ color: Colors.primary }}>{foto ? 'Cambiar foto' : 'Agregar foto'}</Text>
          </TouchableOpacity>

          <Label text="Fecha de Nacimiento" />
          <TouchableOpacity onPress={() => setMostrarPicker(true)} activeOpacity={0.8}>
            <CustomInput
              placeholder="AAAA-MM-DD"
              value={fechaNacimiento}
              editable={false}
              pointerEvents="none"
            />
          </TouchableOpacity>
          {mostrarPicker && (
            <DateTimePicker
              value={fechaDate}
              mode="date"
              display="default"
              onChange={handleDateChange}
              maximumDate={new Date()}
              minimumDate={new Date(1900, 0, 1)}
            />
          )}

          <Label text="Usuario" />
          <CustomInput placeholder="Nombre de usuario" value={usuario} onChangeText={setUsuario} />

          <Label text="Correo Electrónico" />
          <CustomInput placeholder="ejemplo@correo.com" value={correo} onChangeText={setCorreo} />

          <Label text="Contraseña" />
          <View style={GlobalStyles.inputWrapper}>
            <CustomInput
              placeholder="••••••••"
              value={contraseña}
              onChangeText={setContraseña}
              secureTextEntry={!mostrarContraseña}
              style={{ paddingRight: 40 }}
            />
            <TouchableOpacity
              style={GlobalStyles.eyeIcon}
              onPress={() => setMostrarContraseña((prev) => !prev)}
            >
              <Ionicons name={mostrarContraseña ? 'eye' : 'eye-off'} size={24} color={Colors.primary} />
            </TouchableOpacity>
          </View>

          <Label text="Confirmar Contraseña" />
          <View style={GlobalStyles.inputWrapper}>
            <CustomInput
              placeholder="••••••••"
              value={confirmarContraseña}
              onChangeText={setConfirmarContraseña}
              secureTextEntry={!mostrarConfirmarContraseña}
              style={{ paddingRight: 40 }}
            />
            <TouchableOpacity
              style={GlobalStyles.eyeIcon}
              onPress={() => setMostrarConfirmarContraseña((prev) => !prev)}
            >
              <Ionicons name={mostrarConfirmarContraseña ? 'eye' : 'eye-off'} size={24} color={Colors.primary} />
            </TouchableOpacity>
          </View>

          <TouchableOpacity
            onPress={() => {
              if (!aceptaTerminos) handleOpenTerminos();
              else setAceptaTerminos(false);
            }}
            style={{ flexDirection: 'row', alignItems: 'center', marginVertical: 16 }}
          >
            <View
              style={{
                width: 24,
                height: 24,
                borderWidth: 2,
                borderColor: Colors.primary,
                backgroundColor: aceptaTerminos ? Colors.primary : '#fff',
                marginRight: 8,
                justifyContent: 'center',
                alignItems: 'center',
                borderRadius: 6,
              }}
            >
              {aceptaTerminos && <Ionicons name="checkmark" size={18} color="#fff" />}
            </View>
            <View style={{ flex: 1, flexDirection: 'row', flexWrap: 'wrap', alignItems: 'center' }}>
              <Text style={{ color: Colors.primary }}>Acepto los </Text>
              <Text
                style={{ color: Colors.primary, textDecorationLine: 'underline' }}
                onPress={handleOpenTerminos}
              >
                términos y condiciones
              </Text>
            </View>
          </TouchableOpacity>

          {loading ? (
            <ActivityIndicator size="large" color={Colors.primary} />
          ) : (
            <PrimaryButton title="Crear Cuenta" onPress={handleSignup} />
          )}

          <View style={{ alignItems: 'center', marginTop: 24 }}>
            <LinkText
              text="¿Ya tienes una cuenta? Iniciar Sesión"
              onPress={() => navigation.navigate('Login')}
              style={{ marginBottom: 8 }}
            />
          </View>
        </ScrollView>
      </KeyboardAvoidingView>
    </Layout>
  );
};

export default Signup;
