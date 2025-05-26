import React, { useRef, useState } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, ActivityIndicator } from 'react-native';
import { Camera } from 'expo-camera';
import Layout from '../layout/Layout';
import Label from '../components/CustomLabel';

const BACKEND_URL = 'http://TU_IP_O_DOMINIO:8000/predict'; // Cambia por tu endpoint real

const Translate = () => {
  const [hasPermission, setHasPermission] = useState(null);
  const [translation, setTranslation] = useState('');
  const [loading, setLoading] = useState(false);
  const cameraRef = useRef(null);

  React.useEffect(() => {
    (async () => {
      const { status } = await Camera.requestCameraPermissionsAsync();
      setHasPermission(status === 'granted');
    })();
  }, []);

  const handleCaptureAndPredict = async () => {
    if (!cameraRef.current) return;
    setLoading(true);
    try {
      const photo = await cameraRef.current.takePictureAsync({ base64: true, quality: 0.5 });
      const formData = new FormData();
      formData.append('file', {
        uri: photo.uri,
        name: 'frame.jpg',
        type: 'image/jpeg',
      });

      const res = await fetch(BACKEND_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'multipart/form-data' },
        body: formData,
      });
      const data = await res.json();
      setTranslation(data.prediction || 'Sin resultado');
    } catch (error) {
      setTranslation('Error al conectar con el backend');
    }
    setLoading(false);
  };

  if (hasPermission === null) {
    return <View style={styles.container}><Text>Solicitando permisos de cámara...</Text></View>;
  }
  if (hasPermission === false) {
    return <View style={styles.container}><Text>No se tiene acceso a la cámara.</Text></View>;
  }
console.log('Camera:', Camera);
  return (
    <Layout>
      <View style={styles.container}>
        <Label text="Traducción" style={styles.title} />
        <Camera
          style={styles.camera}
          ref={cameraRef}
          type={Camera.Constants.Type.front}
        />
        <TouchableOpacity style={styles.button} onPress={handleCaptureAndPredict} disabled={loading}>
          <Text style={styles.buttonText}>{loading ? 'Procesando...' : 'Detectar Seña'}</Text>
        </TouchableOpacity>
        <View style={styles.translationBox}>
          <Text style={styles.translationText}>{translation}</Text>
        </View>
      </View>
    </Layout>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, padding: 24, justifyContent: 'flex-start' },
  title: { fontSize: 24, color: '#00ffe7', fontWeight: 'bold', marginBottom: 16 },
  camera: {
    width: '100%',
    height: 320,
    borderRadius: 16,
    overflow: 'hidden',
    marginBottom: 16,
    backgroundColor: '#232b3b',
    alignSelf: 'center',
    justifyContent: 'center',
  },
  button: {
    backgroundColor: '#00ffe7',
    borderRadius: 8,
    paddingVertical: 14,
    alignItems: 'center',
    marginBottom: 16,
  },
  buttonText: {
    color: '#181f2b',
    fontWeight: 'bold',
    fontSize: 17,
  },
  translationBox: {
    minHeight: 80,
    backgroundColor: '#232b3b',
    borderRadius: 12,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 16,
    borderWidth: 1,
    borderColor: '#00ffe7',
  },
  translationText: {
    color: '#fff',
    fontSize: 18,
    textAlign: 'center',
  },
});

export default Translate;