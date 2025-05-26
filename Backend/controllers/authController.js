import supabase from '../configs/supabaseClient.js';
import bcrypt from 'bcrypt';
import jwt from 'jsonwebtoken';
import crypto from 'crypto';

// Manejo de errores globales
process.on('uncaughtException', (err) => {
    console.error('Unhandled error:', err);
});

process.on('unhandledRejection', (err) => {
    console.error('Unhandled promise rejection:', err);
});

// Endpoint para registrar usuarios
export const register = async (req, res) => {
    const { name, username, email, password, role } = req.body;

    try {
        // Verificar si la conexión a Supabase es exitosa
        const { data: testData, error: testError } = await supabase.from('users').select('*').limit(1);
        if (testError) {
            console.error('Error en la conexión con Supabase:', testError);
            return res.status(500).json({ error: 'Error de conexión con la base de datos' });
        }

        // Validación de campos obligatorios
        if (!email || !password || !role) {
            return res.status(400).json({ error: 'El correo, la contraseña y el rol son obligatorios' });
        }

        // Validación básica de formato de correo
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        if (!emailRegex.test(email)) {
            return res.status(400).json({ error: 'Correo electrónico inválido' });
        }

        // Validación de longitud de la contraseña
        if (password.length < 6) {
            return res.status(400).json({ error: 'La contraseña debe tener al menos 6 caracteres' });
        }

        // Encriptación de la contraseña
        const saltRounds = 10;
        const passwordHash = await bcrypt.hash(password, saltRounds);

        // Generación de token de verificación
        const verificationToken = crypto.randomBytes(32).toString('hex');

        // Inserción de usuario en la base de datos
        const { data, error } = await supabase
            .from('users')  // Asegúrate de que el nombre de la tabla sea correcto
            .insert([
                {
                    name,
                    username,
                    email,
                    password_hash: passwordHash,
                    role,
                    email_verified: false,  // Inicializamos el correo como no verificado
                    verification_token: verificationToken,
                    reset_token: null,  // Si no se usa, lo dejamos como null
                    reset_token_expires_at: null,  // Inicializamos con null
                    last_login: null,  // Inicializamos con null
                    created_at: new Date(),  // Fecha actual
                    updated_at: new Date(),  // Fecha actual
                },
            ]);

        // Verifica si hay un error
        if (error) {
            console.error('Error al insertar el usuario en la base de datos:', error);
            return res.status(500).json({
                error: `Hubo un error al registrar el usuario: ${error.message || error.details || error.code || 'desconocido'}`
            });
        }

        // Respuesta exitosa
        res.json({
            message: 'Usuario registrado. Verifica tu correo electrónico.',
            verificationToken,  // Puedes enviar el token para que lo uses en el sistema de verificación de correo
        });

    } catch (error) {
        // Respuesta de error con el mensaje específico
        console.error('Error interno:', error);  // Log para depuración
        res.status(500).json({ error: 'Ocurrió un error interno' });
    }
};
//Endpoint de verificacion de correo 
export const verifyemail = async (req, res) => {
    const { verificationToken } = req.body;

    try {
        // Validar que el token de verificación esté presente
        if (!verificationToken) {
            return res.status(400).json({ error: 'Se requiere un token de verificación' });
        }

        // Buscar el usuario con el token de verificación
        const { data, error } = await supabase
            .from('users')  // Asegúrate de que sea el esquema y tabla correctos
            .select('*')
            .eq('verification_token', verificationToken)
            .single();  // Utilizamos single() para obtener solo un resultado

        if (error) {
            console.error('Error al verificar el correo electrónico:', error);
            return res.status(500).json({ error: 'Hubo un error al verificar el correo electrónico' });
        }

        if (!data) {
            return res.status(404).json({ error: 'Token de verificación inválido o expirado' });
        }

        // Si se encuentra el usuario, actualizamos el campo email_verified
        const { error: updateError } = await supabase
            .from('users')
            .update({ email_verified: true })
            .eq('id', data.id);  // Actualizamos el usuario por su ID

        if (updateError) {
            console.error('Error al actualizar el estado de verificación del correo electrónico:', updateError);
            return res.status(500).json({ error: 'Hubo un error al verificar el correo electrónico' });
        }

        // Respuesta exitosa
        res.json({
            message: 'Correo electrónico verificado exitosamente',
        });
        

    } catch (error) {
        console.error('Error interno:', error);
        res.status(500).json({ error: 'Ocurrió un error interno' });
    }
};


// Endpoint para iniciar sesión (login)
export const login = async (req, res) => {
    const { email, password } = req.body;

    try {
        if (!email || !password) {
            return res.status(400).json({ error: 'El correo y la contraseña son obligatorios' });
        }

        const { data, error } = await supabase
            .from('users')
            .select('id, email, password_hash, role, email_verified')
            .eq('email', email)
            .single();

        if (error || !data) {
            console.error('Error en la consulta a la base de datos:', error);
            return res.status(400).json({ error: 'Correo o contraseña incorrectos' });
        }

        if (!data.email_verified) {
            return res.status(400).json({ error: 'El correo electrónico no ha sido verificado' });
        }

        const passwordMatch = await bcrypt.compare(password, data.password_hash);
        if (!passwordMatch) {
            return res.status(400).json({ error: 'Correo o contraseña incorrectos' });
        }

        const token = jwt.sign(
            { userId: data.id, email: data.email, role: data.role },
            process.env.JWT_SECRET,
            { expiresIn: '1h' }
        );

        res.json({
            message: 'Inicio de sesión exitoso',
            token,
        });

    } catch (error) {
        console.error('Error en login:', error);
        res.status(500).json({ error: 'Ocurrió un error al iniciar sesión' });
    }
};

// Endpoint para solicitar restablecer contraseña (envío del token)
export const requestpasswordreset = async (req, res) => {
    const { email } = req.body;

    // Validar formato de correo electrónico
    if (!email || !/\S+@\S+\.\S+/.test(email)) {
        return res.status(400).json({ error: 'Correo inválido' });
    }

    try {
        // Verificar si el correo existe en la base de datos
        const { data, error } = await supabase
            .from('users')
            .select('id, email')
            .eq('email', email)
            .single();

        if (error || !data) {
            return res.status(404).json({ error: 'Correo no encontrado' });
        }

        // Generar un token de restablecimiento
        const resetToken = crypto.randomBytes(32).toString('hex');

        // Definir la expiración del token (por ejemplo, 1 hora)
        const expiresAt = new Date();
        expiresAt.setHours(expiresAt.getHours() + 1);  // 1 hora de expiración

        // Revisar los valores antes de realizar la actualización
        console.log('Token:', resetToken);
        console.log('Expiración del token:', expiresAt);

        // Guardar el token y su fecha de expiración en la base de datos
        const { error: updateError } = await supabase
            .from('users')
            .update({
                reset_token: resetToken,
                reset_token_expires_at: expiresAt,
            })
            .eq('id', data.id);

        // Si hubo un error al guardar el token
        if (updateError) {
            console.error('Error al guardar el token:', updateError);
            return res.status(500).json({ error: 'Error al guardar el token' });
        }

        // Verificación post-actualización
        const { data: updatedUser, error: selectError } = await supabase
            .from('users')
            .select('id, reset_token, reset_token_expires_at')
            .eq('id', data.id)
            .single();

        if (selectError) {
            console.error('Error al verificar los cambios:', selectError);
            return res.status(500).json({ error: 'Error al verificar los cambios del token' });
        }

        console.log('Usuario actualizado con el token:', updatedUser);

        // Responder al cliente con el mensaje
        res.json({
            message: 'Se ha enviado un correo para restablecer tu contraseña. El token de restablecimiento es válido por 1 hora.',
            resetToken, // Enviar el token en la respuesta solo para pruebas.
        });
        
    } catch (error) {
        console.error('Error al solicitar el restablecimiento de contraseña:', error);
      res.status(500).json({ error: 'Ocurrió un error interno' });
    }
};

// Endpoint para restablecer la contraseña
export const resetpassword = async (req, res) => {
    const { resetToken, newPassword } = req.body;

    // Verificar que el token y la nueva contraseña estén presentes
    if (!resetToken || !newPassword) {
        return res.status(400).json({ error: 'Token de restablecimiento y nueva contraseña son obligatorios' });
    }

    try {
        // Buscar el usuario con el token de restablecimiento
        const { data, error } = await supabase
            .from('users')
            .select('id, reset_token, reset_token_expires_at')
            .eq('reset_token', resetToken)
            .single();

        if (error || !data) {
            return res.status(404).json({ error: 'Token inválido o expirado' });
        }

        // Verificar si el token ha expirado
        const currentDate = new Date();
        if (currentDate > new Date(data.reset_token_expires_at)) {
            return res.status(400).json({ error: 'El token ha expirado' });
        }

        // Encriptar la nueva contraseña
        const saltRounds = 10;
        const hashedPassword = await bcrypt.hash(newPassword, saltRounds);

        // Actualizar la contraseña en la base de datos
        const { error: updateError } = await supabase
            .from('users')
            .update({
                password_hash: hashedPassword,
                reset_token: null,  // Limpiar el token
                reset_token_expires_at: null,  // Limpiar la fecha de expiración
            })
            .eq('id', data.id);

        if (updateError) {
            return res.status(500).json({ error: 'Error al restablecer la contraseña' });
        }

        // Respuesta exitosa
        res.json({
            message: 'Contraseña restablecida exitosamente',
        });
        
    } catch (error) {
        console.error('Error al restablecer la contraseña:', error);
        res.status(500).json({ error: 'Ocurrió un error interno' });
    }
};