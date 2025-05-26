import React, { createContext, useState, useEffect } from 'react';
import AsyncStorage from '@react-native-async-storage/async-storage';

export const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadUser = async () => {
      try {
        const storedData = await AsyncStorage.getItem('userData');
        if (storedData) {
          const parsed = JSON.parse(storedData);
          setUser(parsed.user);
        }
      } catch (error) {
        console.error('Error cargando datos de usuario:', error);
      }
      setLoading(false);
    };
    loadUser();
  }, []);

  const login = async (usuario, contrasena) => {
    try {
      const res = await fetch('http://192.168.1.69:3000/api/auth/login',
      //const res = await fetch('http://192.168.1.81:3000/api/auth/login', 
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ usuario, contrasena }),
      });

      const data = await res.json();

      if (!res.ok) {
        return { success: false, error: data.error || 'Error en login' };
      }

      await AsyncStorage.setItem('userData', JSON.stringify({
        token: data.token,
        user: data.user,
      }));

      setUser(data.user);
      return { success: true };
    } catch (error) {
      console.error('Error en login:', error);
      return { success: false, error: 'Error en la conexión' };
    }
  };

  const logout = async () => {
    setUser(null);
    try {
      await AsyncStorage.removeItem('userData');
    } catch (error) {
      console.error('Error en logout:', error);
    }
  };

  return (
    <AuthContext.Provider value={{ user, login, logout, loading }}>
      {children}
    </AuthContext.Provider>
  );
};


