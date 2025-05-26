import React from 'react';
import {
  SafeAreaView,
  View,
  StatusBar,
  StyleSheet,
  Platform,
} from 'react-native';
import Colors from '../static/colors';

const Layout = ({ children }) => {
  return (
    <View style={styles.root}>
      <StatusBar
        barStyle="light-content"
        backgroundColor={Colors.dark}
        translucent
      />
      <SafeAreaView style={styles.safeArea}>
        <View style={styles.container}>{children}</View>
      </SafeAreaView>
    </View>
  );
};

const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: Colors.dark,
  },
  safeArea: {
    flex: 1,
  },
  container: {
    flex: 1,
    paddingHorizontal: 24,
    paddingTop: Platform.OS === 'android' ? StatusBar.currentHeight + 24 : 24,
    backgroundColor: Colors.dark,
  },
});

export default Layout;
