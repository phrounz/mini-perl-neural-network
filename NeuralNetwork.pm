#!/usr/bin/perl

use strict;
use warnings;
use Math::Trig;

my $F_LEARNING_RATE = 1.0;

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

package Util;
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

package Neuron;
	sub new($) {
		my ($rlo_left_layer_neurons) = @_;
		my $self = { f_data => undef, rl_weights => [], rlo_left_layer_neurons => $rlo_left_layer_neurons, f_bias => 0 };
		if (defined $rlo_left_layer_neurons) {
			for (my $i = 0; $i < scalar(@$rlo_left_layer_neurons); ++$i) {
				push @{$self->{rl_weights}}, rand();
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
	sub setWeight($$$) {
		my ($self, $index_neuron, $new_value) = @_;
		my $rl_weights = $self->{rl_weights};
		die $index_neuron." ".scalar(@$rl_weights) if ($index_neuron >= scalar(@$rl_weights));
		$$rl_weights[$index_neuron] = $new_value;
	}
	sub setData($$) {
		my ($self, $f_new_data) = @_;
		$self->{f_data} = $f_new_data;
	}
	sub getError($$) {
		my ($self, $expected_value) = @_;
		return ($self->{f_data} - $expected_value);# ** 2;
	}
	sub compute($$) {
		my ($self, $is_within_last_layer) = @_;
		my $f_sum = 0;
    die unless defined $self->{rlo_left_layer_neurons};
		my $rlo_left_layer_neurons = $self->{rlo_left_layer_neurons};
		for (my $i = 0; $i < scalar(@$rlo_left_layer_neurons); ++$i) {
			$f_sum += ${$self->{rl_weights}}[$i] * $$rlo_left_layer_neurons[$i]->getData();
		}
		$self->{f_data} = ($is_within_last_layer ? Util::sigmoid($f_sum, 0) : Util::sigmoid($f_sum, 0)) / scalar(@$rlo_left_layer_neurons);
    #print "$f_sum $self->{f_data}\n";# check activation function
	}
  sub getDebugInfoStr($) {
    my ($self) = @_;
    my @l_weights_readable = map { sprintf("%.2f",$_) } @{$self->{rl_weights}};
    my $prev_str_info = (defined($self->{rlo_left_layer_neurons})?"(linked)":"(not linked)");
    my $str_weights_readable = join(" ",@l_weights_readable);
    $str_weights_readable = (length($str_weights_readable)>150?substr($str_weights_readable,0,150)."...":$str_weights_readable);
    return "  $self->{f_data} ($str_weights_readable) $prev_str_info\n";
  }

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

package Layer;
	sub new($$) {
		my ($ro_left_layer, $nb_neurons) = @_;
		my $self = { rlo_neurons => [] };
		for (my $i = 0; $i < $nb_neurons; ++$i) {
			my $rlo_left_neurons = (defined($ro_left_layer)?$ro_left_layer->{rlo_neurons}:undef);
			push @{$self->{rlo_neurons}}, Neuron::new($rlo_left_neurons);
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
			$$rlo_neurons[$i]->setData($$rl_data[$i]);
		}
	}
  sub getDebugInfoStr($) { # print all values and weights
    my ($self) = @_;
    return join(" ", map { $_->getDebugInfoStr() } @{$self->{rlo_neurons}});
  }
	# sub getCost($$) {
	# 	my ($self, $rl_expected_result) = @_;
	# 	die unless (scalar(@$rl_expected_result)==scalar(@{$self->{rlo_neurons}}));
	# 	my @l_data = $self->getNeuronsData();
	# 	my $cost = 0.0;
	# 	for (my $i = 0; $i < scalar(@l_data); $i++) {
	# 		$cost += ($l_data[$i] - $$rl_expected_result[$i]) ** 2;
	# 	}
	# 	return $cost;
	# }
	sub getNeuronsData($) {
		my ($self) = @_;
		return map { $_->getData() } @{$self->{rlo_neurons}};
	}

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

package NeuralNetwork;
	sub new($) {
		my $li_nb_neurons_per_layer = shift;
		my $self = { rlo_layers => [], ro_src_image_data_layer => Layer::new(undef, $$li_nb_neurons_per_layer[0] ) };
		my $ro_left_layer = $self->{ro_src_image_data_layer};
		for (my $i = 1; $i < scalar(@$li_nb_neurons_per_layer); ++$i) {
			my $ro_layer = Layer::new( $ro_left_layer, $$li_nb_neurons_per_layer[$i] );
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
    print "Src layer :\n"
      .$self->{ro_src_image_data_layer}->getDebugInfoStr()
      ."\n\n";
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
		my $rlo_layers = $self->{rlo_layers};
		return $$rlo_layers[$index_layer];
	}
	# sub getCost($$) {
	# 	my ($self, $rl_expected_result) = @_;
	# 	return $self->getLayer($self->nbLayers()-1)->getCost($rl_expected_result);
	# }
	sub backpropagate($$) {
		my ($self, $rl_expected_result) = @_;

		my @l_expected_result_this_layer = @$rl_expected_result;
		my @l_expected_result_prev_layer = ();

		for (my $k = $self->nbLayers()-1; $k >= 1; $k--) {

			my $rlo_left_layer = $self->getLayer($k-1);
			my $rlo_this_layer = $self->getLayer($k-0);

			my @l_error_each_neuron;
			for (my $i = 0; $i < $rlo_left_layer->nbNeurons(); ++$i) {
				my $output_data_i = $rlo_left_layer->getNeuron($i)->getData();

				my $sum_errors = 0.0;
				for (my $j = 0; $j < $rlo_this_layer->nbNeurons(); ++$j) {
					my $ro_neuron = $rlo_this_layer->getNeuron($j);

					my $weight_i_j = $ro_neuron->getWeight($i);
					my $output_error_j = $ro_neuron->getError($l_expected_result_this_layer[$j]);
					$sum_errors += ($output_error_j * $weight_i_j);

					my $weight_delta_i_j = $output_data_i * $output_error_j;
					$ro_neuron->setWeight( $i, $weight_i_j - $weight_delta_i_j * $F_LEARNING_RATE);#<-- modify
				}

        my $is_last_layer = ($k == $self->nbLayers()-1 ? 1 : 0);
				my $hidden_error = $sum_errors * ($is_last_layer ? Util::sigmoid($output_data_i, 1) : Util::sigmoid($output_data_i, 1));
				my $error_this_neuron = $hidden_error / $rlo_this_layer->nbNeurons();

				push @l_expected_result_prev_layer, $rlo_left_layer->getNeuron($i)->getError($error_this_neuron);
			}
			@l_expected_result_this_layer = @l_expected_result_prev_layer;
			@l_expected_result_prev_layer = ();
		}
	}

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
1;
