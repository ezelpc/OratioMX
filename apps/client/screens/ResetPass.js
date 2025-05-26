import React, { useState } from 'react';
import { View, Text, TextInput, TouchableOpacity, Alert, ActivityIndicator } from 'react-native';
import Layout from '../layout/Layout';
import Label from '../components/CustomLabel';
import GlobalStyles from '../static/global';
import Colors from '../static/colors';

const ResetPass = ({ navigation }) => {
  const [email, setEmail] = useState('');
  const [loading, setLoading] = useState(false);

  const handleReset = async () => {
    if (!email.trim()) {
      Alert.alert('Campo requerido', 'Por favor ingresa tu correo electrónico.');
      return;
    }
    setLoading(true);
    try {
      const response = await fetch('http://192.168.1.69:3000/api/auth/reset-password', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ correo: email.trim() }),
      });
      const data = await response.json();
      if (response.ok) {
        Alert.alert('Éxito', data.message || 'Revisa tu correo para restablecer tu contraseña.', [
          { text: 'OK', onPress: () => navigation.goBack() },
        ]);
      } else {
        Alert.alert('Error', data.error || data.message || 'No se pudo enviar el correo.');
      }
    } catch (error) {
      Alert.alert('Error', 'No se pudo conectar al servidor.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Layout>
      <View style={[GlobalStyles.container, { justifyContent: 'center', paddingTop: 40 }]}>
        <Label text="Restablecer contraseña" style={GlobalStyles.title} />
        <Text style={[GlobalStyles.text, { marginBottom: 18 }]}>
          Ingresa tu correo electrónico y te enviaremos instrucciones para restablecer tu contraseña.
        </Text>
        <View style={{ marginBottom: 18 }}>
          <View style={GlobalStyles.inputWrapper}>
            <TouchableOpacity activeOpacity={1}>
              <Text
                style={{
                  position: 'absolute',
                  left: 18,
                  top: 14,
                  color: Colors.primary,
                  fontWeight: 'bold',
                  zIndex: 1,
                  opacity: 0.7,
                  fontSize: 15,
                }}
              >
                {/* Puedes poner un ícono aquí si lo deseas */}
              </Text>
              <TextInput
                style={{
                  ...GlobalStyles.input,
                  backgroundColor: '#fff',
                  borderRadius: 24,
                  borderWidth: 1,
                  borderColor: Colors.primary,
                  color: Colors.text,
                  fontSize: 16,
                  paddingLeft: 18,
                }}
                placeholder="Correo electrónico"
                placeholderTextColor="#aaa"
                value={email}
                onChangeText={setEmail}
                keyboardType="email-address"
                autoCapitalize="none"
              />
            </TouchableOpacity>
          </View>
        </View>
        {loading ? (
          <ActivityIndicator color={Colors.primary} style={{ marginTop: 20 }} />
        ) : (
          <TouchableOpacity style={GlobalStyles.button} onPress={handleReset}>
            <Text style={GlobalStyles.buttonText}>Enviar</Text>
          </TouchableOpacity>
        )}
      </View>
    </Layout>
  );
};

export default ResetPass;