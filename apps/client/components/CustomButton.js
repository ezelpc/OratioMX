import React from 'react';
import { Text, TouchableOpacity } from 'react-native';
import GlobalStyles from '../static/global'; // Sin llaves porque es un export default

const CustomButton = ({ title, onPress }) => (
  <TouchableOpacity
    style={GlobalStyles.buttonPrimary} // Cambiado de 'button' a 'buttonPrimary'
    onPress={onPress}
    activeOpacity={0.7}
  >
    <Text style={GlobalStyles.buttonText}>{title}</Text>
  </TouchableOpacity>
);

export default CustomButton;
