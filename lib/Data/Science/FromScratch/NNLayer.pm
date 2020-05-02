package Data::Science::FromScratch::NNLayer;

use Data::Science::FromScratch;
use Moo;
use strictures 2;
use namespace::clean;

=head1 SYNOPSIS

  use Data::Science::FromScratch::Layer;

  my $layer = Data::Science::FromScratch::Layer->new;

=head1 DESCRIPTION

A C<Data::Science::FromScratch::Layer> is an abstract class for building neural networks.

=head1 ATTRIBUTES

=head2 ds

  $obj = $layer->ds;

C<Data::Science::FromScratch> object

=cut

has ds => (
    is        => 'ro',
    lazy      => 1,
    init_args => undef,
    default   => sub { Data::Science::FromScratch->new },
);

=head1 METHODS

=head2 forward

  $v = $layer->forward($input);

=cut

sub forward {
    my ($self, $input) = @_;
}

=head2 backward

  $v = $layer->backward($gradient);

=cut

sub backward {
    my ($self, $gradient) = @_;
}

=head2 params

  $v = $layer->params;

=cut

sub params {
    my ($self) = @_;
    return ();
}

=head2 grads

  $v = $layer->grads;

=cut

sub grads {
    my ($self) = @_;
    return ();
}

1;