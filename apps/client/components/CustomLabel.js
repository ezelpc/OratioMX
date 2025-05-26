import React from 'react';
import { Text } from 'react-native';
import GlobalStyles from '../static/global'; // Corrección: sin llaves

const Label = ({ text, style, numberOfLines, ellipsizeMode }) => (
  <Text
    style={[GlobalStyles.label, style]}
    numberOfLines={numberOfLines}
    ellipsizeMode={ellipsizeMode}
  >
    {text}
  </Text>
);

export default Label;
