import { StyleSheet } from 'react-native';
import Colors from '../static/colors'; // Ajusta la ruta si es necesario

const GlobalStyles = StyleSheet.create({
  // Contenedor principal
  container: {
    flex: 1,
    backgroundColor: 'transparent', // o '#181f2b' si así es tu layout
    padding: 24,
  },
  // Títulos y textos
  title: {
    fontSize: 26,
    color: '#00ffe7',
    fontWeight: 'bold',
    letterSpacing: 1,
    marginBottom: 16,
    textAlign: 'center',
  },
  subtitle: {
    fontSize: 20,
    fontWeight: '600',
    color: Colors.secondary,
    marginBottom: 10,
    textAlign: 'center',
  },
  paragraph: {
    fontSize: 16,
    color: Colors.light,
    lineHeight: 22,
    textAlign: 'center',
  },
  label: {
    fontSize: 14,
    color: Colors.light,
    marginBottom: 6,
  },
  text: {
    color: '#fff',
    fontSize: 16,
    marginBottom: 8,
    textAlign: 'center',
  },
  // Inputs
  input: {
    width: '100%',
    backgroundColor: '#181f2b',
    color: '#fff',
    borderRadius: 8,
    paddingVertical: 12,
    paddingHorizontal: 14,
    fontSize: 16,
    borderWidth: 1.5,
    borderColor: '#00ffe7',
    marginBottom: 16,
  },
  inputWrapper: {
    position: 'relative',
    justifyContent: 'center',
  },
  eyeIcon: {
    position: 'absolute',
    right: 16,
    top: 14,
  },
  inputRow: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    marginTop: 8,
  },
  // Botones
  button: {
    backgroundColor: '#00ffe7',
    borderRadius: 8,
    paddingVertical: 14,
    alignItems: 'center',
    marginTop: 18,
  },
  buttonPrimary: {
    backgroundColor: Colors.primary,
    paddingVertical: 14,
    paddingHorizontal: 28,
    borderRadius: 16,
    alignItems: 'center',
    shadowColor: Colors.accent,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.5,
    shadowRadius: 6,
    elevation: 6,
    marginBottom: 10,
  },
  buttonText: {
    color: '#181f2b',
    fontWeight: 'bold',
    fontSize: 17,
  },
  // Links
  link: {
    color: Colors.secondary,
    textDecorationLine: 'underline',
    fontSize: 14,
    textAlign: 'center',
  },
  // Comentarios
  comentarioBox: {
    backgroundColor: '#232b3b',
    borderRadius: 8,
    padding: 10,
    marginBottom: 8,
  },
  comentarioUsuario: {
    color: '#00ffe7',
    fontWeight: 'bold',
    marginBottom: 2,
  },
  comentarioTexto: {
    color: '#fff',
    fontSize: 15,
  },
  botonEnviar: {
    backgroundColor: '#00ffe7',
    borderRadius: 8,
    paddingVertical: 10,
    paddingHorizontal: 16,
    alignItems: 'center',
    justifyContent: 'center',
  },
  botonEnviarTexto: {
    color: '#181f2b',
    fontWeight: 'bold',
    fontSize: 15,
  },
  comentInputRow: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    justifyContent: 'center',
    padding: 4,
    marginTop: 8,
    marginBottom: 8,
    maxWidth: 600, // <-- Aumenta este valor para permitir más ancho total
    width: '100%',
    alignSelf: 'center',
  },
  comentInputBox: {
    flex: 1,
    backgroundColor: '#fff',
    color: '#181f2b',
    borderRadius: 24,
    paddingVertical: 8,
    paddingHorizontal: 16,
    fontSize: 16,
    borderWidth: 0,
    marginRight: 8,
    minHeight: 40,
    maxHeight: 100,
    maxWidth: 540, // <-- Aumenta este valor para más ancho horizontal
  },
  comentSendButton: {
    backgroundColor: '#00ffe7',
    borderRadius: 24,
    width: 44,
    height: 44,
    alignItems: 'center',
    justifyContent: 'center',
    elevation: 2,
  },
  // Avatar
  avatar: {
    width: 70,           // Tamaño cuadrado
    height: 70,
    borderRadius: 35,    // Mitad del tamaño para que sea redonda
    borderWidth: 2,
    borderColor: '#00ffe7',
    marginBottom: 8,
  },
  avatarPlaceholder: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: '#232b3b',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 8,
    borderWidth: 2,
    borderColor: '#00ffe7',
  },
  // Otros
  contactText: { 
    color: '#000', 
    fontWeight: 'bold', 
    fontSize: 16 
  },
  // FAQ Section
  faqSection: {
    marginTop: 20,
    marginBottom: 24,
  },
  faqItem: {
    backgroundColor: '#232b3b',
    borderRadius: 8,
    padding: 12,
    marginBottom: 12,
  },
  faqQ: {
    color: '#00ffe7',
    fontWeight: 'bold',
    fontSize: 16,
    marginBottom: 4,
  },
  faqA: {
    color: '#fff',
    fontSize: 15,
  },
  contactButton: {
    backgroundColor: '#00ffe7',
    borderRadius: 8,
    paddingVertical: 12,
    alignItems: 'center',
    marginBottom: 24,
  },
  contactText: {
    color: '#181f2b',
    fontWeight: 'bold',
    fontSize: 16,
  },
  // Otros estilos
  headerRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 10,
  },
  menuContainer: {
    position: 'absolute',
    top: 60,
    right: 20,
    width: '70%', // Usa porcentaje, el width real lo puedes pasar inline si lo necesitas
    backgroundColor: '#181f2b',
    borderRadius: 16,
    elevation: 8,
    padding: 18,
    zIndex: 20,
    shadowColor: '#00ffe7',
    shadowOpacity: 0.2,
    shadowRadius: 10,
  },
  menuHeader: {
    alignItems: 'center',
    marginBottom: 18,
  },
  menuSaludo: {
    fontSize: 16,
    color: '#00ffe7',
  },
  menuNombre: {
    fontSize: 18,
    color: '#fff',
    fontWeight: 'bold',
    marginBottom: 6,
  },
  menuOption: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 10,
    gap: 10,
  },
  menuText: {
    color: 'primary',
    fontSize: 16,
    marginLeft: 8,
  },
  metricsRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
    marginVertical: 24,
    gap: 10,
  },
  metricCard: {
    flexBasis: '48%',
    backgroundColor: '#232b3b',
    borderRadius: 14,
    alignItems: 'center',
    padding: 16,
    marginBottom: 10,
    shadowColor: '#00ffe7',
    shadowOpacity: 0.08,
    shadowRadius: 8,
  },
  metricValue: {
    fontSize: 22,
    color: '#00ffe7',
    fontWeight: 'bold',
  },
  metricLabel: {
    fontSize: 14,
    color: '#fff',
    opacity: 0.7,
    textAlign: 'center',
  },
  translateButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#00ffe7',
    borderRadius: 12,
    paddingVertical: 14,
    marginHorizontal: 24,
    marginBottom: 12,
    marginTop: 0,
    shadowColor: '#00ffe7',
    shadowOpacity: 0.15,
    shadowRadius: 8,
    elevation: 4,
    gap: 10,
  },
  translateButtonText: {
    color: '#181f2b',
    fontWeight: 'bold',
    fontSize: 18,
    marginLeft: 8,
    letterSpacing: 1,
  },
  welcomeMsg: {
    fontSize: 16,
    color: '#fff',
    textAlign: 'center',
    marginTop: 16,
  },
  optionRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginVertical: 14,
  },
  optionText: {
    color: '#fff',
    fontSize: 17,
  },
  languageButton: {
    backgroundColor: '#232b3b',
    paddingHorizontal: 16,
    paddingVertical: 6,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#00ffe7',
  },
  languageText: {
    color: '#00ffe7',
    fontWeight: 'bold',
  },
});

export default GlobalStyles;
