import React from 'react';
import { View, Text, ScrollView, Linking, TouchableOpacity } from 'react-native';
import Layout from '../layout/Layout';
import Label from '../components/CustomLabel';
import GlobalStyles from '../static/global';
import { useLanguage } from '../context/LanguageContext';

const textos = {
  es: {
    ayuda: 'Ayuda',
    intro: '¿Necesitas ayuda? Aquí encontrarás preguntas frecuentes y soporte.',
    contactar: 'Contactar soporte',
    faqs: [
      {
        pregunta: '¿Cómo cambio mi contraseña?',
        respuesta: 'Ve a Editar Perfil y escribe una nueva contraseña en el campo correspondiente.',
      },
      {
        pregunta: '¿Cómo cambio el idioma de la aplicación?',
        respuesta: 'En Configuración, selecciona la opción de idioma y elige tu preferido.',
      },
      {
        pregunta: '¿No recibo notificaciones?',
        respuesta: 'Asegúrate de tener activadas las notificaciones en Configuración y en los ajustes de tu dispositivo.',
      },
      {
        pregunta: '¿Cómo contacto al soporte?',
        respuesta: 'Puedes escribirnos a soporte@oratiomx.com o usar el botón de contacto abajo.',
      },
    ],
  },
  en: {
    ayuda: 'Help',
    intro: 'Need help? Here you will find frequently asked questions and support.',
    contactar: 'Contact support',
    faqs: [
      {
        pregunta: 'How do I change my password?',
        respuesta: 'Go to Edit Profile and enter a new password in the corresponding field.',
      },
      {
        pregunta: 'How do I change the app language?',
        respuesta: 'In Settings, select the language option and choose your preferred one.',
      },
      {
        pregunta: 'I am not receiving notifications?',
        respuesta: 'Make sure notifications are enabled in Settings and on your device.',
      },
      {
        pregunta: 'How do I contact support?',
        respuesta: 'You can write to soporte@oratiomx.com or use the contact button below.',
      },
    ],
  }
};

const Help = () => {
  const { idioma } = useLanguage();

  return (
    <Layout>
      <ScrollView contentContainerStyle={GlobalStyles.container}>
        <Label text={textos[idioma].ayuda} style={GlobalStyles.title} />
        <Text style={GlobalStyles.text}>
          {textos[idioma].intro}
        </Text>
        <View style={GlobalStyles.faqSection}>
          {textos[idioma].faqs.map((faq, idx) => (
            <View key={idx} style={GlobalStyles.faqItem}>
              <Text style={GlobalStyles.faqQ}>{faq.pregunta}</Text>
              <Text style={GlobalStyles.faqA}>{faq.respuesta}</Text>
            </View>
          ))}
        </View>
        <TouchableOpacity
          style={GlobalStyles.contactButton}
          onPress={() => Linking.openURL('mailto:soporte@oratiomx.com')}
        >
          <Text style={GlobalStyles.contactText}>{textos[idioma].contactar}</Text>
        </TouchableOpacity>
      </ScrollView>
    </Layout>
  );
};

export default Help;