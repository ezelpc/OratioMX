import React, { useEffect, useState, useContext } from 'react';
import { View, Text, TextInput, TouchableOpacity, FlatList, ActivityIndicator, Alert, KeyboardAvoidingView, Platform } from 'react-native';
import Layout from '../layout/Layout';
import Label from '../components/CustomLabel';
import { AuthContext } from '../context/AuthContext';
import GlobalStyles from '../static/global';
import { useLanguage } from '../context/LanguageContext';

const API_URL = 'http://192.168.1.69:3000/api/comentarios';
//const API_URL = 'http://192.168.1.81:3000/api/comentarios';

const textos = {
  es: {
    titulo: 'Comentarios',
    subtitulo: 'Comparte tus sugerencias, dudas o comentarios sobre OratioMX.',
    placeholder: 'Escribe un comentario...',
    enviando: 'Enviando...',
    errorCargar: 'No se pudieron cargar los comentarios',
    errorEnviar: 'No se pudo enviar el comentario',
    errorConexion: 'No se pudo conectar al servidor',
    anonimo: 'Anónimo',
  },
  en: {
    titulo: 'Comments',
    subtitulo: 'Share your suggestions, questions or comments about OratioMX.',
    placeholder: 'Write a comment...',
    enviando: 'Sending...',
    errorCargar: 'Could not load comments',
    errorEnviar: 'Could not send comment',
    errorConexion: 'Could not connect to server',
    anonimo: 'Anonymous',
  }
};

const Coments = () => {
  const { user } = useContext(AuthContext);
  const { idioma } = useLanguage();
  const [comentario, setComentario] = useState('');
  const [comentarios, setComentarios] = useState([]);
  const [loading, setLoading] = useState(true);
  const [enviando, setEnviando] = useState(false);

  useEffect(() => {
    obtenerComentarios();
  }, []);

  const obtenerComentarios = async () => {
    setLoading(true);
    try {
      const res = await fetch(API_URL);
      const data = await res.json();
      setComentarios(data);
    } catch (error) {
      Alert.alert('Error', textos[idioma].errorCargar);
    }
    setLoading(false);
  };

  const enviarComentario = async () => {
    if (comentario.trim() === '') return;
    setEnviando(true);
    try {
      const res = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          texto: comentario.trim(),
          usuario_id: user?.id,
        }),
      });
      if (res.ok) {
        setComentario('');
        obtenerComentarios();
      } else {
        Alert.alert('Error', textos[idioma].errorEnviar);
      }
    } catch (error) {
      Alert.alert('Error', textos[idioma].errorConexion);
    }
    setEnviando(false);
  };

  return (
    <Layout>
      <KeyboardAvoidingView
        style={{ flex: 1 }}
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        keyboardVerticalOffset={Platform.OS === 'ios' ? 80 : 0}
      >
        <View style={GlobalStyles.container}>
          <Label text={textos[idioma].titulo} style={GlobalStyles.title} />
          <Text style={GlobalStyles.text}>
            {textos[idioma].subtitulo}
          </Text>
          {loading ? (
            <ActivityIndicator color="#00ffe7" style={{ marginTop: 20 }} />
          ) : (
            <FlatList
              data={comentarios}
              keyExtractor={item => item.id?.toString() || Math.random().toString()}
              renderItem={({ item }) => (
                <View style={GlobalStyles.comentarioBox}>
                  <Text style={GlobalStyles.comentarioUsuario}>{item.usuario || textos[idioma].anonimo}:</Text>
                  <Text style={GlobalStyles.comentarioTexto}>{item.texto}</Text>
                </View>
              )}
              style={{ marginVertical: 16 }}
              contentContainerStyle={{ paddingBottom: 16 }}
              showsVerticalScrollIndicator={false}
            />
          )}
          <View style={{ alignItems: 'center', width: '100%' }}>
            <View style={GlobalStyles.comentInputRow}>
              <TextInput
                style={GlobalStyles.comentInputBox}
                placeholder={textos[idioma].placeholder}
                placeholderTextColor="#888"
                value={comentario}
                onChangeText={setComentario}
                multiline
              />
              <TouchableOpacity
                style={[
                  GlobalStyles.comentSendButton,
                  enviando && { opacity: 0.6 }
                ]}
                onPress={enviarComentario}
                disabled={enviando}
              >
                <Text style={GlobalStyles.comentSendButtonText}>
                  {enviando ? '...' : '➤'}
                </Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </KeyboardAvoidingView>
    </Layout>
  );
};

export default Coments;