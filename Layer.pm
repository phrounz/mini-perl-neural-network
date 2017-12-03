#!/usr/bin/perl

use strict;
use warnings;
use Data::Dumper;

use lib ".";
use Neuron;

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

package Layer;

sub new($$) {
	my ($ro_left_layer, $nb_neurons) = @_;
	my $self = { rlo_neurons => [] };
	for (my $i = 0; $i < $nb_neurons; ++$i) {
		my $rlo_left_neurons = (defined($ro_left_layer)?$ro_left_layer->{rlo_neurons}:undef);
		push @{$self->{rlo_neurons}}, Neuron::new($rlo_left_neurons, $nb_neurons);
	}
	return bless $self;
}
sub nbNeurons($) {
	return scalar(@{shift()->{rlo_neurons}});
}
sub getNeuron($$) {
	my ($self, $index_neuron) = @_;
	my $rlo_neurons = $self->{rlo_neurons};
	return $$rlo_neurons[$index_neuron];
}
sub compute($$) {
	my ($self, $is_last_layer) = @_;
	my $rlo_neurons = $self->{rlo_neurons};
	for (my $i = 0; $i < scalar(@$rlo_neurons); ++$i) {
		$$rlo_neurons[$i]->compute($is_last_layer);
	}
}
sub fillData($$) {
	my ($self, $rl_data) = @_;
  unless (scalar(@$rl_data) == $self->nbNeurons()) {
    die scalar(@$rl_data)."!=".$self->nbNeurons() ;# size should not change
  }
	my $rlo_neurons = $self->{rlo_neurons};
	for (my $i = 0; $i < scalar(@$rlo_neurons); ++$i) {
		$$rlo_neurons[$i]->setData($$rl_data[$i] * 1.0/256.0);# / scalar(@$rlo_neurons)
	}
}
sub getDebugInfoStr($) { # print all values and weights
  my ($self) = @_;
  return join(" ", map { $_->getDebugInfoStr() } @{$self->{rlo_neurons}});
}
sub getCost($$) {
	my ($self, $rl_expected_result) = @_;
	die unless (scalar(@$rl_expected_result)==scalar(@{$self->{rlo_neurons}}));
	my @l_data = $self->getNeuronsData();
	my $cost = 0.0;
	for (my $i = 0; $i < scalar(@l_data); $i++) {
		$cost += ($l_data[$i] - $$rl_expected_result[$i]) ** 2;
	}
	return $cost;
}
sub getNeuronsData($) {
	return map { $_->getData() } @{shift()->{rlo_neurons}};
}
# sub changeWeights($) {
# 	map { $_->changeWeights() } @{shift()->{rlo_neurons}};
# }

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

1;
