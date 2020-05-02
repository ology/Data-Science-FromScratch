package Data::Science::FromScratch::NNSequential;

use Moo;
use strictures 2;
use Storable qw(dclone);
use namespace::clean;

=head1 SYNOPSIS

  use Data::Science::FromScratch::NNSequential;

  my $seq = Data::Science::FromScratch::NNSequential->new;

=head1 DESCRIPTION

A C<Data::Science::FromScratch::NNSequential> is a class for building neural networks.

=head1 ATTRIBUTES

=head2 layers

  $obj = $seq->layers;

=cut

has layers => (
    is       => 'rw',
    required => 1,
);

=head1 METHODS

=head2 forward

  $v = $seq->forward($input);

=cut

sub forward {
    my ($self, $input) = @_;
    my $in = dclone $input;
    for my $layer (@{ $self->layers }) {
        $in = $layer->forward($in);
    }
    return $in;
}

=head2 backward

  $v = $layer->backward($gradient);

=cut

sub backward {
    my ($self, $gradient) = @_;
    my $grad = dclone $gradient;
    for my $layer (reverse @{ $self->layers }) {
        $grad = $layer->backward($grad);
    }
    return $grad;
}

=head2 params

  $v = $layer->params;

=cut

sub params {
    my ($self) = @_;
    my @params;
    for my $layer (reverse @{ $self->layers }) {
        push @params, @{ $layer->params };
    }
    return \@params;
}

=head2 grads

  $v = $layer->grads;

=cut

sub grads {
    my ($self) = @_;
    my @grads;
    for my $layer (reverse @{ $self->layers }) {
        push @grads, @{ $layer->grads };
    }
    return \@grads;
}

1;

=head1 SEE ALSO

L<Data::Science::FromScratch::NNLayer>

=cut
