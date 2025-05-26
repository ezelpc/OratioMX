import express from 'express';
import { register, login, verifyemail, requestpasswordreset, resetpassword } from '../controllers/authController.js';

const router = express.Router();

router.post('/register', register);
router.post('/login', login);
router.post('/verifyemail', verifyemail);
router.post('/requestpasswordreset', requestpasswordreset);
router.post('/resetpassword', resetpassword);

export default router;



