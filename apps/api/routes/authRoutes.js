import express from 'express';
const router = express.Router();
import { registerUser, loginUser } from '../controllers/authController.js';  // Usando import

// Rutas para registro y login
router.post('/register', registerUser);
router.post('/login', loginUser);

export default router;  // Usando export default
