#!/usr/bin/perl

use strict;
use warnings;
use Data::Dumper;

use lib ".";
use util;

my $F_LEARNING_RATE = 0.2;
# $F_LEARNING_RATE 0.5 with short training?

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

package Neuron;

sub new($$) {
	my ($rlo_left_layer_neurons, $nb_neurons_this_layer) = @_;
	my $self = {
    f_data => undef,
		rl_weights => [],
		#rl_delta_weights => [],
		rlo_left_layer_neurons => $rlo_left_layer_neurons,
		f_bias => 0.0,
		# f_delta_bias => 0.0,
    #nb_for_delta_bias => 0,
    error => 0.0
	 	};
	if (defined $rlo_left_layer_neurons) {
		for (my $i = 0; $i < scalar(@$rlo_left_layer_neurons); ++$i) {
			push @{$self->{rl_weights}}, (rand()*2.0-1.0);# * $nb_neurons_this_layer / (1.0 * scalar(@$rlo_left_layer_neurons));#*2.0-1.0;#
			#push @{$self->{rl_delta_weights}}, 0.0;
		}
	}
	return bless $self;
}
sub getData($) {
	my ($self) = @_;
	return $self->{f_data};
}
sub getWeight($$) {
	my ($self, $index_neuron) = @_;
	my $rl_weights = $self->{rl_weights};
	die $index_neuron." ".scalar(@$rl_weights) if ($index_neuron >= scalar(@$rl_weights));
	return $$rl_weights[$index_neuron];
}
# sub oldAddToDeltaWeight($$$) {
# 	my ($self, $index_neuron, $increment) = @_;
# 	my $rl_delta_weights = $self->{rl_delta_weights};
# 	die $index_neuron." ".scalar(@$rl_delta_weights) if ($index_neuron >= scalar(@$rl_delta_weights));
# 	$$rl_delta_weights[$index_neuron] += $increment;
# }
sub addToWeight($$) {
  my ($self, $increment) = @_;
  my $rl_weights = $self->{rl_weights};
  unless (defined $self->{rlo_left_layer_neurons}) {
    print Data::Dumper::Dumper($self);
    die scalar(@$rl_weights);
  }
	my $rlo_left_layer_neurons = $self->{rlo_left_layer_neurons};
  for (my $i = 0; $i < scalar(@$rl_weights); ++$i) {
    $$rl_weights[$i] += $increment * $F_LEARNING_RATE * $$rlo_left_layer_neurons[$i]->getData();
  }
  $self->{f_bias} = $increment * $F_LEARNING_RATE;
}
sub setData($$) {
	my ($self, $f_new_data) = @_;
	$self->{f_data} = $f_new_data;
}
sub setError($$) {
	my ($self, $error) = @_;
	return $self->{error} = $error;
}
sub getError($) { return shift()->{error}; }
sub compute($$) {
	my ($self, $is_within_last_layer) = @_;
	my $f_sum = 0;
  die unless defined $self->{rlo_left_layer_neurons};
	my $rlo_left_layer_neurons = $self->{rlo_left_layer_neurons};
	for (my $i = 0; $i < scalar(@$rlo_left_layer_neurons); ++$i) {
		$f_sum += ${$self->{rl_weights}}[$i] * $$rlo_left_layer_neurons[$i]->getData();
	}
	$f_sum += $self->{f_bias};
  #print "$f_sum\n";
	#$f_sum /= scalar(@$rlo_left_layer_neurons);

	#$self->{f_delta_bias} += $f_sum;
	#$self->{nb_for_delta_bias}++;

	$self->{f_data} = ($is_within_last_layer ? util::sigmoid($f_sum, 0) : util::sigmoid($f_sum, 0));
  #print "$self->{f_data} ";
	# / scalar(@$rlo_left_layer_neurons);
  #print "$f_sum $self->{f_data}\n";# check activation function
}
sub getDebugInfoStr($) {
  my ($self) = @_;
	#my $prev_str_info = (defined($self->{rlo_left_layer_neurons})?"(linked)":"(not linked)");

  my @l_weights_readable = map { $_>0.01?sprintf("%.2f",$_):sprintf("%.2f",$_*1000.0)."E-3" } @{$self->{rl_weights}};
	#my @l_delta_weights_readable = map { $_>0.01?sprintf("%.2f",$_):sprintf("%.2f",$_*1000.0)."E-3" } @{$self->{rl_delta_weights}};

  my $sum_weights = 0.0;
  map { $sum_weights+= $_ } @{$self->{rl_weights}};
  #my $sum_delta_weights = 0.0;
  #map { $sum_delta_weights+= $_ } @{$self->{rl_delta_weights}};

	my @l_com;
	for (my $i = 0; $i < scalar(@l_weights_readable); ++$i) {
		push @l_com, $l_weights_readable[$i];#."+".$l_delta_weights_readable[$i]."R";
	}
	my $str_weights =join(" ",@l_com);
  $str_weights = (length($str_weights)>120?substr($str_weights,0,120)."...":$str_weights);
  return "  $self->{f_data} ($str_weights) $sum_weights $self->{f_bias}\n";#$sum_delta_weights $prev_str_info$self->{f_bias}
}
# sub changeWeights($) {
# 	my ($self) = @_;
# 	my $rl_weights = $self->{rl_weights};
# 	my $rl_delta_weights =$self->{rl_delta_weights};
# 	for (my $i = 0; $i < scalar(@$rl_weights); ++$i) {
# 		$$rl_weights[$i] += $$rl_delta_weights[$i] * $F_LEARNING_RATE;
# 		$$rl_delta_weights[$i] = 0.0;
# 	}
#
# 	#$self->{f_bias} = $self->{f_delta_bias} * -1.0 / $self->{nb_for_delta_bias};
# 	$self->{f_delta_bias} = 0.0;
# 	$self->{nb_for_delta_bias} = 0;
# }

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

1;
