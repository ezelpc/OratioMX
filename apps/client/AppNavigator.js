import React, { useContext } from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';

import { AuthContext } from './context/AuthContext';

// Screens
import Login from './screens/Login';
import Signup from './screens/Signup';
import Welcome from './screens/Welcome';
import Terminos from './screens/Terminos';
import Settings from './screens/Settings';
import EditProfile from './screens/EditProfile';
import Help from './screens/Help';
import ResetPass from './screens/ResetPass';
import Translate from './screens/Translate';
import Coments from './screens/Coments';

const Stack = createNativeStackNavigator();

export default function AppNavigator() {
  const { user, loading } = useContext(AuthContext);

  if (loading) return null; // Puedes poner un SplashScreen aquí si quieres

  return (
    <NavigationContainer>
      <Stack.Navigator screenOptions={{ headerShown: false }}>
        {!user ? (
          <>
            <Stack.Screen name="Login" component={Login} />
            <Stack.Screen name="Signup" component={Signup} />
            <Stack.Screen name="ResetPass" component={ResetPass} />
          </>
        ) : (
          <>
            <Stack.Screen name="Welcome" component={Welcome} />
            <Stack.Screen name="Terminos" component={Terminos} />
            <Stack.Screen name="Settings" component={Settings} />
            <Stack.Screen name="EditProfile" component={EditProfile} />
            <Stack.Screen name="Help" component={Help} />
            <Stack.Screen name="ResetPass" component={ResetPass} />
            <Stack.Screen name="Translate" component={Translate} />
            <Stack.Screen name="Coments" component={Coments} />
          </>
        )}
      </Stack.Navigator>
    </NavigationContainer>
  );
}
