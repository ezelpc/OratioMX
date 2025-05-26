import React from 'react';
import { TextInput } from 'react-native';
import GlobalStyles from '../static/global'; // Corrección: sin llaves
import Colors from '../static/colors';

const CustomInput = ({
  placeholder,
  value,
  onChangeText,
  secureTextEntry = false,
  style,
  ...rest // Para extender otras props como keyboardType, autoCapitalize, etc.
}) => (
  <TextInput
    style={[GlobalStyles.input, style]} // Permite sobreescribir estilos externos
    placeholder={placeholder}
    placeholderTextColor={Colors.textSecondary}
    value={value}
    onChangeText={onChangeText}
    secureTextEntry={secureTextEntry}
    {...rest}
  />
);

export default CustomInput;
