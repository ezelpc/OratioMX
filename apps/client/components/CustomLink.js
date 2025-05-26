import React from 'react';
import { Text, TouchableOpacity } from 'react-native';
import GlobalStyles from '../static/global';

const LinkText = ({ text, children, onPress, style }) => (
  <TouchableOpacity onPress={onPress} activeOpacity={0.7}>
    <Text style={[GlobalStyles.link, style]}>
      {text ? text : children}
    </Text>
  </TouchableOpacity>
);

export default LinkText;
