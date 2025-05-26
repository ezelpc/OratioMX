import express from 'express';
import dotenv from 'dotenv';
import cors from 'cors';
import authRoutes from './routes/authRoutes.js';
import comentariosRoutes from './routes/comentarios.js';

dotenv.config();

const app = express();

const corsOptions = {
  //origin: 'http://192.168.1.81:3000', // Puedes usar '*' solo para pruebas locales
  origin: '*',
  methods: ['GET', 'POST'],
  allowedHeaders: ['Content-Type', 'Authorization'],
};

app.use(cors(corsOptions));
app.use(express.json());

app.use('/api/auth', authRoutes);
app.use('/api/comentarios', comentariosRoutes);

// Si quieres que este archivo sea ejecutable directamente:


// Si lo usas como módulo en otro archivo, puedes exportar app:
export default app;
