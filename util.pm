#!/usr/bin/perl

use strict;
use warnings;

use Math::Trig;

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

package util;

sub sigmoid($$) {
	my ($x, $is_derivative) = @_;
	return ($x * (1.0 - $x)) if $is_derivative;
	return (1.0 / (1.0 + exp(-1.0 * $x)));
}
sub tanh($$) {
	my ($x, $is_derivative) = @_;
	return (1 - ($x ** 2)) if $is_derivative;
	return Math::Trig::tanh($x);
}
sub reLU($$) {
  my ($x, $is_derivative) = @_;
  return ($x > 0 ? 1 : 0) if $is_derivative;
  # http://kawahara.ca/what-is-the-derivative-of-relu/
  return ($x > 0 ? $x : 0);
}
sub leakyReLU($$) {
  my ($x, $is_derivative) = @_;
  return ($x > 0 ? 1.0 : -0.1) if $is_derivative;
  return ($x > 0 ? $x : -0.1 * $x);
}
sub leakyCustomReLU($$) {
  my ($x, $is_derivative) = @_;
  return ($x > 0 ? 0.0001 : -0.00001) if $is_derivative;
  return ($x > 0 ? 0.0001 * $x : -0.00001 * $x);
}

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
1;
