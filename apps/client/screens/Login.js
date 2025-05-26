import React, { useState } from 'react';
import { View, TouchableOpacity, Alert, ActivityIndicator, Text } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { useAuth } from '../hooks/useAuth';
import Layout from '../layout/Layout';
import CustomInput from '../components/CustomInput';
import PrimaryButton from '../components/CustomButton';
import Label from '../components/CustomLabel';
import Ionicons from 'react-native-vector-icons/Ionicons';
import LinkText from '../components/CustomLink';
import GlobalStyles from '../static/global';
import Colors from '../static/colors';

const Login = ({ navigation }) => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [mostrarContraseña, setMostrarContraseña] = useState(false);
  const [loading, setLoading] = useState(false);
  const [recordar, setRecordar] = useState(false);

  const { login } = useAuth();

  const handleLogin = async () => {
    if (!email || !password) {
      Alert.alert('Campos requeridos', 'Por favor ingresa usuario/correo y contraseña.');
      return;
    }

    setLoading(true);
    const result = await login(email, password);

    if (result.success) {
      if (recordar) {
        await AsyncStorage.setItem('user', JSON.stringify(email));
      }
    } else {
      Alert.alert('Error', result.error || 'Error en login');
    }
    setLoading(false);
  };

  return (
    <Layout>
      <View style={[GlobalStyles.container, { justifyContent: 'center', paddingTop: 40 }]}>
        <Label text="Usuario o correo electrónico" />
        <CustomInput
          placeholder="usuario o correo"
          value={email}
          onChangeText={setEmail}
          keyboardType="email-address"
          autoCapitalize="none"
          style={{
            backgroundColor: '#fff',
            borderRadius: 24,
            borderWidth: 1,
            borderColor: Colors.primary,
            color: Colors.text,
            fontSize: 16,
            marginBottom: 12,
          }}
        />
        <Label text="Contraseña" />
        <View style={{ position: 'relative', justifyContent: 'center' }}>
          <CustomInput
            placeholder="••••••••"
            value={password}
            onChangeText={setPassword}
            secureTextEntry={!mostrarContraseña}
            style={{
              backgroundColor: '#fff',
              borderRadius: 24,
              borderWidth: 1,
              borderColor: Colors.primary,
              color: Colors.text,
              fontSize: 16,
              marginBottom: 12,
              paddingRight: 44,
            }}
          />
          <TouchableOpacity
            style={{
              position: 'absolute',
              right: 14,
              top: '50%',
              transform: [{ translateY: -16 }],
              zIndex: 2,
            }}
            onPress={() => setMostrarContraseña(!mostrarContraseña)}
          >
            <Ionicons
              name={mostrarContraseña ? 'eye' : 'eye-off'}
              size={24}
              color={Colors.primary}
            />
          </TouchableOpacity>
        </View>

        <TouchableOpacity
          onPress={() => setRecordar(!recordar)}
          style={{
            flexDirection: 'row',
            alignItems: 'center',
            marginTop: 8,
            marginBottom: 16,
            alignSelf: 'flex-start',
          }}
          activeOpacity={0.7}
        >
          <View
            style={{
              width: 22,
              height: 22,
              borderWidth: 2,
              borderColor: Colors.primary,
              marginRight: 10,
              justifyContent: 'center',
              alignItems: 'center',
              borderRadius: 6,
              backgroundColor: recordar ? Colors.primary : '#fff',
            }}
          >
            {recordar && <Ionicons name="checkmark" size={16} color="#fff" />}
          </View>
          <Text style={{ color: Colors.primary, fontSize: 16 }}>Recordar usuario</Text>
        </TouchableOpacity>

        {loading ? (
          <ActivityIndicator size="large" color={Colors.primary} style={{ marginVertical: 20, alignSelf: 'center' }} />
        ) : (
          <PrimaryButton title="Iniciar sesión" onPress={handleLogin} />
        )}

        <View style={{ marginTop: 28, alignItems: 'center' }}>
          <LinkText onPress={() => navigation.navigate('ResetPass')}>
            ¿Olvidaste tu contraseña?
          </LinkText>
          <LinkText onPress={() => navigation.navigate('Signup')}>
            ¿No tienes cuenta? Regístrate
          </LinkText>
        </View>
      </View>
    </Layout>
  );
};

export default Login;
