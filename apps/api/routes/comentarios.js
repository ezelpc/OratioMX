import express from 'express';
import { obtenerComentarios, crearComentario } from '../controllers/comentariosController.js';

const router = express.Router();

router.get('/', obtenerComentarios);
router.post('/', crearComentario);

export default router;