import React from 'react';
import { View, Text, ScrollView, StyleSheet, TouchableOpacity, Alert } from 'react-native';
import Layout from '../layout/Layout';
import Ionicons from 'react-native-vector-icons/Ionicons';

const Terminos = ({ navigation, route }) => {
  const { onAccept } = route.params || {};

  const handleAccept = () => {
    if (onAccept && typeof onAccept === 'function') {
      onAccept(true); // llamar la función callback
    } else {
      // Si no hay callback, enviamos params a la pantalla anterior
      navigation.navigate({
        name: 'Signup',
        params: { acepta: true },
        merge: true,
      });
    }

    Alert.alert('Gracias', 'Has aceptado los términos y condiciones.');
    navigation.goBack();
  };
  return (
    <Layout>
      <View style={styles.header}>
        <TouchableOpacity onPress={() => navigation.goBack()} style={styles.backButton}>
          <Ionicons name="arrow-back" size={24} color="#007bff" />
        </TouchableOpacity>
        <Text style={styles.title}>Términos y Condiciones</Text>
      </View>

      <ScrollView contentContainerStyle={styles.container}>
        <Text style={styles.text}>
          Bienvenido a nuestra aplicación Oratio MX, un traductor de lenguaje de señas mexicana que utiliza visión por computadora para ofrecer traducciones en tiempo real. Al usar esta aplicación, aceptas los siguientes términos y condiciones:
          {'\n\n'}
          1. Uso del Servicio{'\n'}
          Oratio MX está diseñada para ayudar en la comunicación entre personas sordomudas y oyentes, y debe usarse conforme a la ley y con respeto hacia todos los usuarios.
          {'\n\n'}
          2. Privacidad y Tratamiento de Datos{'\n'}
          Recopilamos datos técnicos y de uso para mejorar continuamente el modelo de traducción, incluyendo la posibilidad de reaprendizaje automático basado en el uso que hacen los usuarios. No compartiremos información personal con terceros sin tu consentimiento explícito.
          {'\n\n'}
          3. Responsabilidades{'\n'}
          Oratio MX se proporciona “tal cual” y no garantizamos la precisión absoluta de las traducciones. No nos hacemos responsables de daños o malentendidos derivados del uso inapropiado de la aplicación.
          {'\n\n'}
          4. Propiedad Intelectual{'\n'}
          El contenido, modelo y código fuente de Oratio MX son propiedad exclusiva de los desarrolladores y están protegidos por leyes de propiedad intelectual.
          {'\n\n'}
          5. Cambios en los Términos{'\n'}
          Nos reservamos el derecho de modificar estos términos en cualquier momento. Te notificaremos oportunamente para que puedas revisar los cambios.
          {'\n\n'}
          6. Consentimiento{'\n'}
          Al continuar usando Oratio MX, confirmas que has leído, entendido y aceptado estos términos y condiciones.
          {'\n\n'}
          Gracias por confiar en Oratio MX para mejorar la comunicación inclusiva.
        </Text>

        <TouchableOpacity style={styles.button} onPress={handleAccept}>
          <Text style={styles.buttonText}>Aceptar Términos y Condiciones</Text>
        </TouchableOpacity>
      </ScrollView>
    </Layout>
  );
};


const styles = StyleSheet.create({
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 12,
    paddingHorizontal: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#ddd',
  },
  backButton: {
    marginRight: 12,
    padding: 4,
  },
  title: {
    fontSize: 22,
    fontWeight: 'bold',
    color: '#28a745',
  },
  container: {
    padding: 16,
    paddingBottom: 40,
  },
  text: {
    fontSize: 16,
    color: '#ddd',
    lineHeight: 24,
    marginBottom: 24,
  },
  button: {
    backgroundColor: '#28a745',
    paddingVertical: 14,
    borderRadius: 8,
    alignItems: 'center',
    marginTop: 10,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
    elevation: 5,
  },
  buttonText: {
    color: '#fff',
    fontWeight: '600',
    fontSize: 16,
  },
});

export default Terminos;

