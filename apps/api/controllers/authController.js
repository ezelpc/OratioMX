import bcrypt from 'bcryptjs';
import jwt from 'jsonwebtoken';
import supabase from '../DB/supabaseClient.js';
import { v4 as uuidv4 } from 'uuid';

// Middleware de autenticación para verificar el token JWT
const authenticate = (req, res, next) => {
  const token = req.header('Authorization')?.replace('Bearer ', '');
  
  // Si no hay token en la cabecera, se retorna error
  if (!token) {
    return res.status(401).json({ error: 'No autorizado. Token no proporcionado.' });
  }

  try {
    // Verificación del token
    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    req.user = decoded; // Guarda los datos del usuario en la request
    next();
  } catch (error) {
    // En caso de error en la verificación
    return res.status(401).json({ error: 'Token inválido o expirado' });
  }
};

// Registro de usuario
const registerUser = async (req, res) => {
  const {
    nombres, primer_apellido, segundo_apellido, fecha_nacimiento,
    usuario, correo_electronico, contrasena, rol, foto
  } = req.body;

  try {
    // Verificación de campos obligatorios
    if (!usuario?.trim() || !contrasena?.trim() || !correo_electronico?.trim()) {
      return res.status(400).json({ error: 'Campos obligatorios faltantes' });
    }

    // Encriptar la contraseña
    const hashedPassword = await bcrypt.hash(contrasena, 10);

    // Insertar el nuevo usuario en la base de datos
    const { data, error } = await supabase
      .from('usuarios')
      .insert([{
        id: uuidv4(),
        nombres,
        primer_apellido,
        segundo_apellido,
        fecha_nacimiento,
        usuario,
        correo_electronico,
        contrasena: hashedPassword,
        rol: rol || 'Usuario',
        foto
      }]);


    // Verificación de errores de inserción
    if (error || !data) {
      console.error('Error de Supabase:', error);
      throw new Error(error?.message || 'Error al insertar el usuario');
    }

    // Respuesta exitosa
    res.status(201).json({
      message: 'Usuario registrado con éxito',
      user: {
        id: data[0].id,
        usuario: data[0].usuario,
        rol: data[0].rol
      }
    });
  } catch (error) {
    // Manejo de errores
    console.error(error);
    res.status(500).json({ error: error.message || 'Error interno en el servidor' });
  }
};

// Login de usuario (por usuario o correo electrónico)
const loginUser = async (req, res) => {
  const { usuario, contrasena } = req.body;

  try {
    // Buscar al usuario por nombre de usuario o correo electrónico
    const { data: user, error } = await supabase
      .from('usuarios')
      .select('*')
      .or(`usuario.eq.${usuario},correo_electronico.eq.${usuario}`) // <-- Cambiado aquí
      .single();

    // Verificar si el usuario existe
    if (error || !user) {
      return res.status(404).json({ error: 'Usuario o correo electrónico no encontrado' });
    }

    // Comparar la contraseña ingresada con la almacenada
    const isMatch = await bcrypt.compare(contrasena, user.contrasena);
    if (!isMatch) return res.status(401).json({ error: 'Contraseña incorrecta' });

    // Actualizar el último inicio de sesión
    const { error: updateError } = await supabase
      .from('usuarios')
      .update({ last_login: new Date().toISOString() })
      .eq('id', user.id);

    if (updateError) {
      return res.status(500).json({ error: 'No se pudo actualizar el último inicio de sesión' });
    }

    // Crear un token JWT para el usuario
    const token = jwt.sign(
      {
        id: user.id,
        usuario: user.usuario,
        rol: user.rol
      },
      process.env.JWT_SECRET,
      { expiresIn: process.env.JWT_EXPIRATION || '3h' }
    );

    // Respuesta exitosa
    res.json({
      message: 'Login exitoso',
      token,
      user: {
        id: user.id,
        nombres: user.nombres,
        usuario: user.usuario,
        rol: user.rol
      }
    });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: error.message || 'Error interno en el servidor' });
  }
};

// Exportar las funciones como funciones predeterminadas
export { authenticate, registerUser, loginUser };
