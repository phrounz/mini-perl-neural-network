#!/usr/bin/perl

use strict;
use warnings;
use Data::Dumper;

use lib ".";
use util;
use Layer;

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

package NeuralNetwork;

sub new($) {
	my $li_nb_neurons_per_layer = shift;
	my $self = { rlo_layers => [], ro_src_image_data_layer => Layer::new(undef, $$li_nb_neurons_per_layer[0] ) };#, -1
	my $ro_left_layer = $self->{ro_src_image_data_layer};
	for (my $i = 1; $i < scalar(@$li_nb_neurons_per_layer); ++$i) {
		my $ro_layer = Layer::new( $ro_left_layer, $$li_nb_neurons_per_layer[$i] );#, $i-1
		push @{$self->{rlo_layers}}, $ro_layer;
		$ro_left_layer = $ro_layer;
	}
	return bless $self;
}
sub setSrcImageData($$) {
	my ($self, $rl_src_image_data) = @_;
	$self->{ro_src_image_data_layer}->fillData($rl_src_image_data);
}
sub setSrcImageDataRaw($$) {
	my ($self, $src_image_data_raw) = @_;
  my @l_src_image_data = map { ord($_) } (split '', $src_image_data_raw);
  $self->setSrcImageData(\@l_src_image_data);
}
sub compute($) {
	my ($self) = @_;
	my $rlo_layers = $self->{rlo_layers};
	for (my $i = 0; $i < $self->nbLayers(); ++$i) {
		$$rlo_layers[$i]->compute($i==$self->nbLayers()-1);
	}
}
sub getResult($) {
	my ($self) = @_;
	my $rlo_layers = $self->{rlo_layers};
	return $self->getLayer($self->nbLayers()-1)->getNeuronsData();
}
sub printResultStr($) {
  print "(".join(",", map {sprintf("%.3f",$_)} shift()->getResult()).")\n";
}
sub printComparedResultsStr($$) {
  my ($self, $rl_expected_result) = @_;
  print "(".join(",", $rl_expected_result).") => "
		."(".join(",", map {sprintf("%.3f",$_)} shift()->getResult()).")"." cost="
		.$self->getCost($rl_expected_result)."\n";
}
sub printDebug($) {
  my ($self) = @_;
  my $rlo_layers = $self->{rlo_layers};
	print "===================\n";
  # print "Src layer :\n"
  #   .$self->{ro_src_image_data_layer}->getDebugInfoStr()
  #   ."\n\n";
  for (my $i = 0; $i < $self->nbLayers(); ++$i) {
    print "Layer $i :\n"
      .$$rlo_layers[$i]->getDebugInfoStr()
      ."\n\n";
	}
}

sub nbLayers($) {
	return scalar(@{shift()->{rlo_layers}});
}
sub getNbNeuronsLastLayer($) {
  my ($self) = @_;
  return $self->getLayer($self->nbLayers()-1)->nbNeurons();
}
sub getLayer($$) {
	my ($self, $index_layer) = @_;
	return $self->{ro_src_image_data_layer} if ($index_layer == -1);
	my $rlo_layers = $self->{rlo_layers};
	return $$rlo_layers[$index_layer];
}
sub getCost($$) {
	my ($self, $rl_expected_result) = @_;
	return $self->getLayer($self->nbLayers()-1)->getCost($rl_expected_result);
}
sub backpropagate($$) {
	my ($self, $rl_expected_result) = @_;

 	my $rlo_last_layer = $self->getLayer($self->nbLayers()-1);
	for (my $i = 0; $i < $rlo_last_layer->nbNeurons(); ++$i) {
		#_network[_network.size() - 1][i].error = results[i] * (1 - results[i]) * (expected[i] - results[i]);
		#results[i] * (1 - results[i]) * (expected[i] - results[i]);
		my $ro_neuron = $rlo_last_layer->getNeuron($i);
		#my $error = util::sigmoid($$rl_expected_result[$i] - $ro_neuron->getData(), 1);
		my $error = util::sigmoid($ro_neuron->getData(), 1) * ($$rl_expected_result[$i] - $ro_neuron->getData());
		$ro_neuron->setError($error);
		$ro_neuron->addToWeight($error);
	}

	# my @l_expected_result_this_layer = @$rl_expected_result;
	# my @l_expected_result_prev_layer = ();

	for (my $k = $self->nbLayers()-1; $k >= 1; $k--) {

		my $rlo_left_layer = $self->getLayer($k-1);
		my $rlo_this_layer = $self->getLayer($k-0);

		my @l_error_each_neuron;
		for (my $i = 0; $i < $rlo_left_layer->nbNeurons(); ++$i) {
			my $ro_neuron_i = $rlo_left_layer->getNeuron($i);
			my $output_data_i = $ro_neuron_i->getData();

			my $sum_errors_j = 0.0;
			for (my $j = 0; $j < $rlo_this_layer->nbNeurons(); ++$j) {
				my $ro_neuron_j = $rlo_this_layer->getNeuron($j);

				# my $weight_i_j = $ro_neuron->getWeight($i);

				#my $error = $ro_neuron->getData() * (1.0 - $ro_neuron->getData()) * ($ro_neuron->getData() - $$rl_expected_result[$i]);

				# my $output_error_j = $ro_neuron->getError($l_expected_result_this_layer[$j]);
				# $sum_errors_j += ($output_error_j * $weight_i_j);
				$sum_errors_j += $ro_neuron_j->getError() * $ro_neuron_j->getWeight($i);

				#my $weight_delta_i_j = $output_data_i * $output_error_j;

				#$ro_neuron->addToDeltaWeight( $i, $weight_delta_i_j * -1.0);

				# print " $weight_delta_i_j $i " if ($k==0 && $weight_delta_i_j!=0 && $j==0);
# if ($k==0 ){
# 	 print " $weight_delta_i_j $i " if ($k==0 && $weight_delta_i_j!=0 && $j==0);
# 	my @l1= map { $_>0.01?sprintf("%.2f",$_):sprintf("%.2f",$_*1000.0)."E-03" } @{$ro_neuron->{rl_delta_weights}};
# 	print "=> ".join("", @l1)."\n";
# }
				#print "".$ro_neuron->getData()." - $l_expected_result_this_layer[$j] => $output_error_j (*$output_data_i=>$weight_delta_i_j) \n" if ($k==0 && $output_data_i!=0);
			}

			#print "===>\n" if ($k==0);
			#print Data::Dumper::Dumper($rlo_this_layer) if ($k==0);
			#die if ($k==0);
			#$ro_neuron_i->addToWeight($sum_errors_j);
			my $error2 = util::sigmoid($ro_neuron_i->getData(), 1) * $sum_errors_j;
			$ro_neuron_i->setError($error2);
			$ro_neuron_i->addToWeight($error2);

			# if ($k==0) {
			# 	print "===>".$rlo_this_layer->getDebugInfoStr()."\n";
			# }

			#my $is_last_layer = ($k == $self->nbLayers()-1 ? 1 : 0);
			# my $hidden_error = $sum_errors_j * ($is_last_layer ? util::sigmoid($output_data_i, 1) : util::sigmoid($output_data_i, 1));
			# my $error_this_neuron = $hidden_error;# / $rlo_this_layer->nbNeurons();

			# #print "==> $sum_errors_j $error_this_neuron ".$rlo_left_layer->getNeuron($i)->getData()."\n" if ($k==0);# && $rlo_left_layer->getNeuron($i)->getData()!=0.0);
			# push @l_expected_result_prev_layer, $rlo_left_layer->getNeuron($i)->getError($error_this_neuron);
		}
		# @l_expected_result_this_layer = @l_expected_result_prev_layer;
		# @l_expected_result_prev_layer = ();
		# if ($k==0) {
		#  	print "===>".$rlo_this_layer->getDebugInfoStr()."\n";
		# }
	}

}
# sub changeWeights($) {
# 	foreach my $ro_layer (@{shift()->{rlo_layers}}) {
# 		$ro_layer->changeWeights();
# 	}
# }

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
1;
