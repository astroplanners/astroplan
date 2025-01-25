# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for scheduling observations.
"""

import copy
from abc import ABCMeta, abstractmethod

import numpy as np

from astropy import units as u
from astropy.time import Time
from astropy.table import Table

from .utils import time_grid_from_range, stride_array
from .constraints import AltitudeConstraint
from .target import get_skycoord

__all__ = ['ObservingBlock', 'TransitionBlock', 'Schedule', 'Slot',
           'Scheduler', 'SequentialScheduler', 'PriorityScheduler',
           'Transitioner', 'Scorer']


class ObservingBlock:
    """
    An observation to be scheduled, consisting of a target and associated
    constraints on observations.
    """
    @u.quantity_input(duration=u.second)
    def __init__(self, target, duration, priority, configuration={}, constraints=None, name=None):
        """
        Parameters
        ----------
        target : `~astroplan.FixedTarget`
            Target to observe

        duration : `~astropy.units.Quantity`
            exposure time

        priority : integer or float
            priority of this object in the target list. 1 is highest priority,
            no maximum

        configuration : dict
            Configuration metadata

        constraints : list of `~astroplan.constraints.Constraint` objects
            The constraints to apply to this particular observing block.  Note
            that constraints applicable to the entire list should go into the
            scheduler.

        name : integer or str
            User-defined name or ID.

        """
        self.target = target
        self.duration = duration
        self.priority = priority
        self.configuration = configuration
        self.constraints = constraints
        self.name = name
        self.start_time = self.end_time = None
        self.observer = None

    def __repr__(self):
        orig_repr = object.__repr__(self)
        if self.start_time is None or self.end_time is None:
            return orig_repr.replace('object at',
                                     '({0}, unscheduled) at'
                                     .format(self.target.name))
        else:
            s = '({0}, {1} to {2}) at'.format(self.target.name, self.start_time.iso,
                                              self.end_time.iso)
            return orig_repr.replace('object at', s)

    @property
    def constraints_scores(self):
        if not (self.start_time and self.duration):
            return None
        # TODO: setup a way of caching or defining it as an attribute during scheduling
        elif self.observer:
            return {constraint: constraint(self.observer, self.target,
                                           times=[self.start_time, self.start_time + self.duration])
                    for constraint in self.constraints}

    @classmethod
    def from_exposures(cls, target, priority, time_per_exposure,
                       number_exposures, readout_time=0 * u.second,
                       configuration={}, constraints=None):
        duration = number_exposures * (time_per_exposure + readout_time)
        ob = cls(target, duration, priority, configuration, constraints)
        ob.time_per_exposure = time_per_exposure
        ob.number_exposures = number_exposures
        ob.readout_time = readout_time
        return ob


class Scorer:
    """
    Returns scores and score arrays from the evaluation of constraints on
    observing blocks
    """

    def __init__(self, blocks, observer, schedule, global_constraints=[]):
        """
        Parameters
        ----------
        blocks : list of `~astroplan.scheduling.ObservingBlock` objects
            list of blocks that need to be scored
        observer : `~astroplan.Observer`
            the observer
        schedule : `~astroplan.scheduling.Schedule`
            The schedule inside which the blocks should fit
        global_constraints : list of `~astroplan.Constraint` objects
            any ``Constraint`` that applies to all the blocks
        """
        self.blocks = blocks
        self.observer = observer
        self.schedule = schedule
        self.global_constraints = global_constraints
        self.targets = get_skycoord([block.target for block in self.blocks])

    def create_score_array(self, time_resolution=1*u.minute):
        """
        this makes a score array over the entire schedule for all of the
        blocks and each `~astroplan.Constraint` in the .constraints of
        each block and in self.global_constraints.

        Parameters
        ----------
        time_resolution : `~astropy.units.Quantity`
            the time between each scored time

        Returns
        -------
        score_array : `~numpy.ndarray`
            array with dimensions (# of blocks, schedule length/ ``time_resolution``
        """
        start = self.schedule.start_time
        end = self.schedule.end_time
        times = time_grid_from_range((start, end), time_resolution)
        score_array = np.ones((len(self.blocks), len(times)))
        for i, block in enumerate(self.blocks):
            # TODO: change the default constraints from None to []
            if block.constraints:
                for constraint in block.constraints:
                    applied_score = constraint(self.observer, block.target,
                                               times=times)
                    score_array[i] *= applied_score
        for constraint in self.global_constraints:
            score_array *= constraint(self.observer, self.targets, times,
                                      grid_times_targets=True)
        return score_array

    @classmethod
    def from_start_end(cls, blocks, observer, start_time, end_time,
                       global_constraints=[]):
        """
        for if you don't have a schedule/ aren't inside a scheduler
        """
        dummy_schedule = Schedule(start_time, end_time)
        sc = cls(blocks, observer, dummy_schedule, global_constraints)
        return sc


class TransitionBlock:
    """
    Parameterizes the "dead time", e.g. between observations, while the
    telescope is slewing, instrument is reconfiguring, etc.
    """

    def __init__(self, components, start_time=None):
        """
        Parameters
        ----------
        components : dict
            A dictionary mapping the reason for an observation's dead time to
            `~astropy.units.Quantity` objects with time units

        start_time : `~astropy.units.Quantity`
            Start time of observation
        """
        self._components = None
        self.duration = None
        self.start_time = start_time
        self.components = components

    def __repr__(self):
        orig_repr = object.__repr__(self)
        comp_info = ', '.join(['{0}: {1}'.format(c, t)
                               for c, t in self.components.items()])
        if self.start_time is None or self.end_time is None:
            return orig_repr.replace('object at', ' ({0}, unscheduled) at'.format(comp_info))
        else:
            s = '({0}, {1} to {2}) at'.format(comp_info, self.start_time.iso, self.end_time.iso)
            return orig_repr.replace('object at', s)

    @property
    def end_time(self):
        return self.start_time + self.duration

    @property
    def components(self):
        return self._components

    @components.setter
    def components(self, val):
        duration = 0*u.second
        for t in val.values():
            duration += t

        self._components = val
        self.duration = duration

    @classmethod
    @u.quantity_input(duration=u.second)
    def from_duration(cls, duration):
        # for testing how to put transitions between observations during
        # scheduling without considering the complexities of duration
        tb = TransitionBlock({'duration': duration})
        return tb


class Schedule:
    """
    An object that represents a schedule, consisting of a list of
    `~astroplan.scheduling.Slot` objects.
    """
    # as currently written, there should be no consecutive unoccupied slots
    # this should change to allow for more flexibility (e.g. dark slots, grey slots)

    def __init__(self, start_time, end_time, constraints=None):
        """
        Parameters
        ----------
        start_time : `~astropy.time.Time`
            The starting time of the schedule; the start of your
            observing window.
        end_time : `~astropy.time.Time`
           The ending time of the schedule; the end of your
           observing window
        constraints : sequence of `~astroplan.constraints.Constraint` s
           these are constraints that apply to the entire schedule
        """
        self.start_time = start_time
        self.end_time = end_time
        self.slots = [Slot(start_time, end_time)]
        self.observer = None

    def __repr__(self):
        return ('Schedule containing ' + str(len(self.observing_blocks)) +
                ' observing blocks between ' + str(self.slots[0].start.iso) +
                ' and ' + str(self.slots[-1].end.iso))

    @property
    def observing_blocks(self):
        return [slot.block for slot in self.slots if isinstance(slot.block, ObservingBlock)]

    @property
    def scheduled_blocks(self):
        return [slot.block for slot in self.slots if slot.block]

    @property
    def open_slots(self):
        return [slot for slot in self.slots if not slot.occupied]

    def to_table(self, show_transitions=True, show_unused=False):
        # TODO: allow different coordinate types
        target_names = []
        start_times = []
        end_times = []
        durations = []
        ra = []
        dec = []
        config = []
        for slot in self.slots:
            if hasattr(slot.block, 'target'):
                start_times.append(slot.start.iso)
                end_times.append(slot.end.iso)
                durations.append(slot.duration.to(u.minute).value)
                target_names.append(slot.block.target.name)
                ra.append(u.Quantity(slot.block.target.ra))
                dec.append(u.Quantity(slot.block.target.dec))
                config.append(slot.block.configuration)
            elif show_transitions and slot.block:
                start_times.append(slot.start.iso)
                end_times.append(slot.end.iso)
                durations.append(slot.duration.to(u.minute).value)
                target_names.append('TransitionBlock')
                ra.append('')
                dec.append('')
                changes = list(slot.block.components.keys())
                if 'slew_time' in changes:
                    changes.remove('slew_time')
                config.append(changes)
            elif slot.block is None and show_unused:
                start_times.append(slot.start.iso)
                end_times.append(slot.end.iso)
                durations.append(slot.duration.to(u.minute).value)
                target_names.append('Unused Time')
                ra.append('')
                dec.append('')
                config.append('')
        return Table([target_names, start_times, end_times, durations, ra, dec, config],
                     names=('target', 'start time (UTC)', 'end time (UTC)',
                            'duration (minutes)', 'ra', 'dec', 'configuration'))

    def new_slots(self, slot_index, start_time, end_time):
        """
        Create new slots by splitting a current slot.

        Parameters
        ----------
        slot_index : int
            The index of the slot to split

        start_time : `~astropy.time.Time`
            The start time for the slot to create

        end_time : `~astropy.time.Time`
            The end time for the slot to create

        Returns
        -------
        new_slots : list of `~astroplan.scheduling.Slot` s
            The new slots created
        """
        # this is intended to be used such that there aren't consecutive unoccupied slots
        new_slots = self.slots[slot_index].split_slot(start_time, end_time)
        return new_slots

    def insert_slot(self, start_time, block):
        """
        Insert a slot into schedule and associate a block to the new slot.

        Parameters
        ----------
        start_time : `~astropy.time.Time`
            The start time for the new slot.
        block : `~astroplan.scheduling.ObservingBlock`
            The observing block to insert into new slot.

        Returns
        -------
        slots : list of `~astroplan.scheduling.Slot` objects
            The new slots in the schedule.
        """
        # due to float representation, this will change block start time
        # and duration by up to 1 second in order to fit in a slot
        for j, slot in enumerate(self.slots):
            if ((slot.start < start_time or abs(slot.start-start_time) < 1*u.second)
                    and (slot.end > start_time + 1*u.second)):
                slot_index = j
        if (block.duration - self.slots[slot_index].duration) > 1*u.second:
            raise ValueError('longer block than slot')
        elif self.slots[slot_index].end - block.duration < start_time:
            start_time = self.slots[slot_index].end - block.duration

        if abs((self.slots[slot_index].duration - block.duration)) < 1 * u.second:
            # slot duration is very similar to block duration.
            # force equality so block fits
            block.duration = self.slots[slot_index].duration
            start_time = self.slots[slot_index].start
            end_time = self.slots[slot_index].end
        elif abs(self.slots[slot_index].start - start_time) < 1*u.second:
            # start time of block is very close to slot start time
            # force equality to avoid tiny gaps
            start_time = self.slots[slot_index].start
            end_time = start_time + block.duration
        elif abs(self.slots[slot_index].end - start_time - block.duration) < 1*u.second:
            # end time is very close to slot end time
            # force equality to avoid tiny gaps
            end_time = self.slots[slot_index].end
        else:
            end_time = start_time + block.duration

        if isinstance(block, ObservingBlock):
            # TODO: make it shift observing/transition blocks to fill small amounts of open space
            block.end_time = start_time+block.duration
        earlier_slots = self.slots[:slot_index]
        later_slots = self.slots[slot_index+1:]
        block.start_time = start_time
        new_slots = self.new_slots(slot_index, start_time, end_time)
        for new_slot in new_slots:
            if new_slot.middle:
                new_slot.occupied = True
                new_slot.block = block
        self.slots = earlier_slots + new_slots + later_slots
        return earlier_slots + new_slots + later_slots

    def change_slot_block(self, slot_index, new_block=None):
        """
        Change the block associated with a slot.

        This is currently designed to work for TransitionBlocks in PriorityScheduler
        The assumption is that the slot afterwards is open and that the start time
        will remain the same.

        If the block is changed to None, the slot is merged with the slot
        afterwards to make a longer slot.

        Parameters
        ----------
        slot_index : int
            The slot to edit
        new_block : `~astroplan.scheduling.TransitionBlock`, default None
            The new transition block to insert in this slot
        """
        if self.slots[slot_index + 1].block:
            raise IndexError('slot afterwards is full')
        if new_block is not None:
            new_end = self.slots[slot_index].start + new_block.duration
            self.slots[slot_index].end = new_end
            self.slots[slot_index].block = new_block
            self.slots[slot_index + 1].start = new_end
            return slot_index
        else:
            self.slots[slot_index + 1].start = self.slots[slot_index].start
            del self.slots[slot_index]
            return slot_index - 1


class Slot:
    """
    A time slot consisting of a start and end time
    """

    def __init__(self, start_time, end_time):
        """
        Parameters
        ----------
        start_time : `~astropy.time.Time`
            The starting time of the slot
        end_time : `~astropy.time.Time`
            The ending time of the slot
        """
        self.start = start_time
        self.end = end_time
        self.occupied = False
        self.middle = False
        self.block = None

    @property
    def duration(self):
        return self.end - self.start

    def split_slot(self, early_time, later_time):
        """
        Split this slot and insert a new one.

        Will return the new slots created, which can either
        be one, two or three slots depending on if there is
        space remaining before or after the inserted slot.

        Parameters
        ----------
        early_time : `~astropy.time.Time`
            The start time of the new slot to insert.
        later_time : `~astropy.time.Time`
            The end time of the new slot to insert.
        """
        # check if the new slot would overwrite occupied/other slots
        if self.occupied:
            raise ValueError('slot is already occupied')

        new_slot = Slot(early_time, later_time)
        new_slot.middle = True
        early_slot = Slot(self.start, early_time)
        late_slot = Slot(later_time, self.end)

        if early_time > self.start and later_time < self.end:
            return [early_slot, new_slot, late_slot]
        elif early_time > self.start:
            return [early_slot, new_slot]
        elif later_time < self.end:
            return [new_slot, late_slot]
        else:
            return [new_slot]


class Scheduler:
    """
    Schedule a set of `~astroplan.scheduling.ObservingBlock` objects
    """

    __metaclass__ = ABCMeta

    @u.quantity_input(gap_time=u.second, time_resolution=u.second)
    def __init__(self, constraints, observer, transitioner=None,
                 gap_time=5*u.min, time_resolution=20*u.second):
        """
        Parameters
        ----------
        constraints : sequence of `~astroplan.constraints.Constraint`
            The constraints to apply to *every* observing block.  Note that
            constraints for specific blocks can go on each block individually.
        observer : `~astroplan.Observer`
            The observer/site to do the scheduling for.
        transitioner : `~astroplan.scheduling.Transitioner` (required)
            The object to use for computing transition times between blocks.
            Leaving it as ``None`` will cause an error.
        gap_time : `~astropy.units.Quantity` with time units
            The maximum length of time a transition between ObservingBlocks
            could take.
        time_resolution : `~astropy.units.Quantity` with time units
            The smallest factor of time used in scheduling, all Blocks scheduled
            will have a duration that is a multiple of it.
        """
        self.constraints = constraints
        self.observer = observer
        self.transitioner = transitioner

        if not hasattr(self.transitioner, '__call__'):
            raise ValueError("A callable Transitioner is required")

        self.gap_time = gap_time
        self.time_resolution = time_resolution

    def __call__(self, blocks, schedule):
        """
        Schedule a set of `~astroplan.scheduling.ObservingBlock` objects.

        Parameters
        ----------
        blocks : list of `~astroplan.scheduling.ObservingBlock` objects
            The observing blocks to schedule.  Note that the input
            `~astroplan.scheduling.ObservingBlock` objects will *not* be
            modified - new ones will be created and returned.
        schedule : `~astroplan.scheduling.Schedule` object
            A schedule that the blocks will be scheduled in. At this time
            the ``schedule`` must be empty, only defined by a start and
            end time.

        Returns
        -------
        schedule : `~astroplan.scheduling.Schedule`
            A schedule objects which consists of `~astroplan.scheduling.Slot`
            objects with and without populated ``block`` objects containing either
            `~astroplan.scheduling.TransitionBlock` or `~astroplan.scheduling.ObservingBlock`
            objects with populated ``start_time`` and ``end_time`` or ``duration`` attributes
        """
        self.schedule = schedule
        self.schedule.observer = self.observer
        # these are *shallow* copies
        copied_blocks = [copy.copy(block) for block in blocks]
        schedule = self._make_schedule(copied_blocks)
        return schedule

    @abstractmethod
    def _make_schedule(self, blocks):
        """
        Does the actual business of scheduling. The ``blocks`` passed in should
        have their ``start_time` and `end_time`` modified to reflect the
        schedule. Any necessary `~astroplan.scheduling.TransitionBlock` should
        also be added.  Then the full set of blocks should be returned as a list
        of blocks, along with a boolean indicating whether or not they have been
        put in order already.

        Parameters
        ----------
        blocks : list of `~astroplan.scheduling.ObservingBlock` objects
            Can be modified as it is already copied by ``__call__``

        Returns
        -------
        schedule : `~astroplan.scheduling.Schedule`
            A schedule objects which consists of `~astroplan.scheduling.Slot`
            objects with and without populated ``block`` objects containing either
            `~astroplan.scheduling.TransitionBlock` or `~astroplan.scheduling.ObservingBlock`
            objects with populated ``start_time`` and ``end_time`` or ``duration`` attributes.
        """
        raise NotImplementedError
        # return schedule

    @classmethod
    @u.quantity_input(duration=u.second)
    def from_timespan(cls, center_time, duration, **kwargs):
        """
        Create a new instance of this class given a center time and duration.

        Parameters
        ----------
        center_time : `~astropy.time.Time`
            Mid-point of time-span to schedule.

        duration : `~astropy.units.Quantity` or `~astropy.time.TimeDelta`
            Duration of time-span to schedule
        """
        start_time = center_time - duration / 2.
        end_time = center_time + duration / 2.
        return cls(start_time, end_time, **kwargs)


class SequentialScheduler(Scheduler):
    """
    A scheduler that does "stupid simple sequential scheduling".  That is, it
    simply looks at all the blocks, picks the best one, schedules it, and then
    moves on.
    """

    def __init__(self, *args, **kwargs):
        super(SequentialScheduler, self).__init__(*args, **kwargs)

    def _make_schedule(self, blocks):
        pre_filled = np.array([[block.start_time, block.end_time] for
                               block in self.schedule.scheduled_blocks])
        if len(pre_filled) == 0:
            a = self.schedule.start_time
            filled_times = Time([a - 1*u.hour, a - 1*u.hour,
                                 a - 1*u.minute, a - 1*u.minute])
            pre_filled = filled_times.reshape((2, 2))
        else:
            filled_times = Time(pre_filled.flatten())
            pre_filled = filled_times.reshape((int(len(filled_times)/2), 2))
        for b in blocks:
            if b.constraints is None:
                b._all_constraints = self.constraints
                b.constraints = [AltitudeConstraint(min=0 * u.deg)]
            else:
                if AltitudeConstraint not in b.constraints:
                    b.constraints.append(AltitudeConstraint(min=0 * u.deg))
                b._all_constraints = self.constraints + b.constraints
            # to make sure the scheduler has some constraint to work off of
            # and to prevent scheduling of targets below the horizon
            # TODO : change default constraints to [] and switch to append
            if b._all_constraints is None:
                b._all_constraints = [AltitudeConstraint(min=0 * u.deg)]
                b.constraints = [AltitudeConstraint(min=0 * u.deg)]
                
            # If there is no constraint on Altititude, specifically giving an AltitudeConstraint
            # for the target to be above horizon.
            elif not any(isinstance(c, AltitudeConstraint) for c in b._all_constraints):
                b._all_constraints.append(AltitudeConstraint(min=0 * u.deg))
                if b.constraints is None:
                    b.constraints = [AltitudeConstraint(min=0 * u.deg)]
                else:
                    b.constraints.append(AltitudeConstraint(min=0 * u.deg))
            b._duration_offsets = u.Quantity([0*u.second, b.duration/2,
                                              b.duration])
            b.observer = self.observer
        current_time = self.schedule.start_time
        while (len(blocks) > 0) and (current_time < self.schedule.end_time):
            # first compute the value of all the constraints for each block
            # given the current starting time
            block_transitions = []
            block_constraint_results = []
            for b in blocks:
                # first figure out the transition
                if len(self.schedule.observing_blocks) > 0:
                    trans = self.transitioner(
                        self.schedule.observing_blocks[-1], b, current_time, self.observer)
                else:
                    trans = None
                block_transitions.append(trans)
                transition_time = 0*u.second if trans is None else trans.duration

                times = current_time + transition_time + b._duration_offsets

                # make sure it isn't in a pre-filled slot
                if (any((current_time < filled_times) & (filled_times < times[2])) or
                        any(abs(pre_filled.T[0]-current_time) < 1*u.second)):
                    block_constraint_results.append(0)

                else:
                    constraint_res = []
                    for constraint in b._all_constraints:
                        constraint_res.append(constraint(
                            self.observer, b.target, times))
                    # take the product over all the constraints *and* times
                    block_constraint_results.append(np.prod(constraint_res))

            # now identify the block that's the best
            bestblock_idx = np.argmax(block_constraint_results)

            if block_constraint_results[bestblock_idx] == 0.:
                # if even the best is unobservable, we need a gap
                current_time += self.gap_time
            else:
                # If there's a best one that's observable, first get its transition
                trans = block_transitions.pop(bestblock_idx)
                if trans is not None:
                    self.schedule.insert_slot(trans.start_time, trans)
                    current_time += trans.duration

                # now assign the block itself times and add it to the schedule
                newb = blocks.pop(bestblock_idx)
                newb.start_time = current_time
                current_time += newb.duration
                newb.end_time = current_time
                newb.constraints_value = block_constraint_results[bestblock_idx]

                self.schedule.insert_slot(newb.start_time, newb)

        return self.schedule


class PriorityScheduler(Scheduler):
    """
    A scheduler that optimizes a prioritized list.  That is, it
    finds the best time for each ObservingBlock, in order of priority.
    """

    def __init__(self, *args, **kwargs):
        """

        """
        super(PriorityScheduler, self).__init__(*args, **kwargs)

    def _get_filled_indices(self, times):
        is_open_time = np.ones(len(times), bool)
        # close times that are already filled
        pre_filled = np.array([[block.start_time, block.end_time] for
                               block in self.schedule.scheduled_blocks if
                               isinstance(block, ObservingBlock)])
        for start_end in pre_filled:
            filled = np.where((start_end[0] < times) & (times < start_end[1]))
            if len(filled[0]) > 0:
                is_open_time[filled[0]] = False
                is_open_time[min(filled[0]) - 1] = False
        return is_open_time

    def _make_schedule(self, blocks):
        # Combine individual constraints with global constraints, and
        # retrieve priorities from each block to define scheduling order

        _all_times = []
        _block_priorities = np.zeros(len(blocks))

        # make sure we don't schedule below the horizon
        if self.constraints is None:
            self.constraints = [AltitudeConstraint(min=0 * u.deg)]
        else:
            self.constraints.append(AltitudeConstraint(min=0 * u.deg))

        for i, b in enumerate(blocks):
            b._duration_offsets = u.Quantity([0 * u.second, b.duration / 2, b.duration])
            _block_priorities[i] = b.priority
            _all_times.append(b.duration)
            b.observer = self.observer

        # Define a master schedule
        # Generate grid of time slots, and a mask for previous observations

        time_resolution = self.time_resolution
        times = time_grid_from_range([self.schedule.start_time, self.schedule.end_time],
                                     time_resolution=time_resolution)

        # generate the score arrays for all of the blocks
        scorer = Scorer(blocks, self.observer, self.schedule,
                        global_constraints=self.constraints)
        score_array = scorer.create_score_array(time_resolution)

        # Sort the list of blocks by priority
        sorted_indices = np.argsort(_block_priorities, kind='mergesort')

        unscheduled_blocks = []
        # Compute the optimal observation time in priority order
        for i in sorted_indices:
            b = blocks[i]
            # Compute possible observing times by combining object constraints
            # with the master open times mask
            constraint_scores = score_array[i]

            # Add up the applied constraints to prioritize the best blocks
            # And then remove any times that are already scheduled
            is_open_time = self._get_filled_indices(times)
            constraint_scores[~is_open_time] = 0

            # Select the most optimal time

            # calculate the number of time slots needed for this exposure
            _stride_by = int(np.ceil(float(b.duration / time_resolution)))

            # Stride the score arrays by that number
            _strided_scores = stride_array(constraint_scores, _stride_by)

            # Collapse the sub-arrays
            # (run them through scorekeeper again? Just add them?
            # If there's a zero anywhere in there, def. have to skip)
            good = np.all(_strided_scores > 1e-5, axis=1)
            sum_scores = np.zeros(len(_strided_scores))
            sum_scores[good] = np.sum(_strided_scores[good], axis=1)

            if np.all(constraint_scores == 0) or np.all(~good):
                # No further calculation if no times meet the constraints
                _is_scheduled = False
            else:
                # schedulable in principle, provided the transition
                # does not prevent us from fitting it in.
                # loop over valid times and see if it fits
                # TODO: speed up by searching multiples of time resolution?
                for idx in np.argsort(-sum_scores, kind='mergesort'):
                    if sum_scores[idx] <= 0.0:
                        # we've run through all optimal blocks
                        _is_scheduled = False
                        break
                    try:
                        start_time_idx = idx
                        new_start_time = times[start_time_idx]
                        # attempt to schedule block
                        _is_scheduled = self.attempt_insert_block(b, new_start_time, start_time_idx)
                        if _is_scheduled:
                            break
                    except IndexError:
                        # idx can extend past end of _strided_open_time
                        _is_scheduled = False
                        break

            if not _is_scheduled:
                unscheduled_blocks.append(b)

        return self.schedule

    def attempt_insert_block(self, b, new_start_time, start_time_idx):
        # set duration to be exact multiple of time resolution
        duration_indices = int(np.floor(
            float(b.duration / self.time_resolution)))
        b.duration = duration_indices * self.time_resolution

        # add 1 second to the start time to allow for scheduling at the start of a slot
        slot_index = [q for q, slot in enumerate(self.schedule.slots)
                      if slot.start < new_start_time + 1*u.second < slot.end][0]
        slots_before = self.schedule.slots[:slot_index]
        slots_after = self.schedule.slots[slot_index + 1:]

        # now check if there's a transition block where we want to go
        # if so, we delete it. A new one will be added if needed
        delete_this_block_first = False
        if self.schedule.slots[slot_index].block:
            if isinstance(self.schedule.slots[slot_index].block, ObservingBlock):
                raise ValueError('block already occupied')
            else:
                delete_this_block_first = True

        # no slots yet, so we should be fine to just shove this in
        if not (slots_before or slots_after):
            b.end_idx = start_time_idx + duration_indices
            b.start_idx = start_time_idx
            if b.constraints is None:
                b.constraints = self.constraints
            elif self.constraints is not None:
                b.constraints = b.constraints + self.constraints
            try:
                self.schedule.insert_slot(new_start_time, b)
                return True
            except ValueError as error:
                # this shouldn't ever happen
                print('Failed to insert {} into schedule.\n{}'.format(
                        b.target.name, str(error)
                      ))
                return False

        # Other slots exist, so now we have to see if it will fit
        # if slots before or after, we need `TransitionBlock`s
        tb_before = None
        tb_before_already_exists = False
        tb_after = None
        if slots_before:
            if isinstance(
                    self.schedule.slots[slot_index - 1].block, ObservingBlock):
                # make a transitionblock
                tb_before = self.transitioner(
                    self.schedule.slots[slot_index - 1].block, b,
                    self.schedule.slots[slot_index - 1].end, self.observer)
            elif isinstance(self.schedule.slots[slot_index - 1].block, TransitionBlock):
                tb_before = self.transitioner(
                    self.schedule.slots[slot_index - 2].block, b,
                    self.schedule.slots[slot_index - 2].end, self.observer)
                tb_before_already_exists = True

        if slots_after:
            slot_offset = 2 if delete_this_block_first else 1
            if isinstance(
                    self.schedule.slots[slot_index + slot_offset].block, ObservingBlock):
                # make a transition object after the new ObservingBlock
                tb_after = self.transitioner(
                    b, self.schedule.slots[slot_index + slot_offset].block,
                    new_start_time + b.duration, self.observer)

        # tweak durations to exact multiple of time resolution
        for block in (tb_before, tb_after):
            if block is not None:
                block.duration = self.time_resolution * int(
                    np.ceil(float(block.duration / self.time_resolution))
                )

        # if we want to shift the OBs to minimise gaps, here is
        # where we should do it.
        # Find the smallest shift (forward or backward) to close gap
        # Check against tolerances (constraints must still be met)
        # Shift if OK and update new_start_time and start_time_idx

        # Now let's see if the block and transition can fit in the schedule
        if slots_before:
            # we're OK if the index at the end of the updated transition
            # is less than or equal to `start_time_idx`
            ob_offset = 2 if tb_before_already_exists else 1
            previous_ob = self.schedule.slots[slot_index - ob_offset]
            if tb_before:
                transition_indices = int(tb_before.duration / self.time_resolution)
            else:
                transition_indices = 0

            if start_time_idx < previous_ob.block.end_idx + transition_indices:
                # cannot schedule
                return False

        if slots_after:
            # we're OK if the index at end of OB (plus transition)
            # is smaller than the start_index of the slot after
            slot_offset = 2 if delete_this_block_first else 1
            next_ob = self.schedule.slots[slot_index + slot_offset].block
            end_idx = start_time_idx + duration_indices
            if tb_after:
                end_idx += int(tb_after.duration / self.time_resolution)
                if end_idx >= next_ob.start_idx:
                    # cannot schedule
                    return False

        # OK, we should be OK to schedule now!
        try:
            # delete this block if it's a TransitionBlock
            if delete_this_block_first:
                slot_index = self.schedule.change_slot_block(slot_index, new_block=None)
            if tb_before and tb_before_already_exists:
                self.schedule.change_slot_block(slot_index - 1, new_block=tb_before)
            elif tb_before:
                self.schedule.insert_slot(tb_before.start_time, tb_before)
            elif tb_before_already_exists and not tb_before:
                # we already have a TB here, but we no longer need it!
                self.schedule.change_slot_block(slot_index-1, new_block=None)

            b.end_idx = start_time_idx + duration_indices
            b.start_idx = start_time_idx
            if b.constraints is None:
                b.constraints = self.constraints
            elif self.constraints is not None:
                b.constraints = b.constraints + self.constraints
            self.schedule.insert_slot(new_start_time, b)

            if tb_after:
                self.schedule.insert_slot(tb_after.start_time, tb_after)

        except ValueError as error:
            # this shouldn't ever happen
            print('Failed to insert {} (dur: {}) into schedule.\n{}\n{}'.format(
                   b.target.name, b.duration, new_start_time.iso, str(error)
                  ))
            return False

        return True


class Transitioner:
    """
    A class that defines how to compute transition times from one block to
    another.
    """
    u.quantity_input(slew_rate=u.deg/u.second)

    def __init__(self, slew_rate=None, instrument_reconfig_times=None):
        """
        Parameters
        ----------
        slew_rate : `~astropy.units.Quantity` with angle/time units
            The slew rate of the telescope
        instrument_reconfig_times : dict of dicts or None
            If not None, gives a mapping from property names to another
            dictionary. The second dictionary maps 2-tuples of states to the
            time it takes to transition between those states (as an
            `~astropy.units.Quantity`), can also take a 'default' key
            mapped to a default transition time.
        """
        self.slew_rate = slew_rate
        self.instrument_reconfig_times = instrument_reconfig_times

    def __call__(self, oldblock, newblock, start_time, observer):
        """
        Determines the amount of time needed to transition from one observing
        block to another.  This uses the parameters defined in
        ``self.instrument_reconfig_times``.

        Parameters
        ----------
        oldblock : `~astroplan.scheduling.ObservingBlock` or None
            The initial configuration/target
        newblock : `~astroplan.scheduling.ObservingBlock` or None
            The new configuration/target to transition to
        start_time : `~astropy.time.Time`
            The time the transition should start
        observer : `astroplan.Observer`
            The observer at the time

        Returns
        -------
        transition : `~astroplan.scheduling.TransitionBlock` or None
            A transition to get from ``oldblock`` to ``newblock`` or `None` if
            no transition is necessary
        """
        components = {}
        if (self.slew_rate is not None and (oldblock is not None) and (newblock is not None)):
            # use the constraints cache for now, but should move that machinery
            # to observer
            from .constraints import _get_altaz
            from .target import get_skycoord
            if oldblock.target != newblock.target:
                targets = get_skycoord([oldblock.target, newblock.target])
                aaz = _get_altaz(start_time, observer, targets)['altaz']
                sep = aaz[0].separation(aaz[1])
                if sep/self.slew_rate > 1 * u.second:
                    components['slew_time'] = sep / self.slew_rate

        if self.instrument_reconfig_times is not None:
            components.update(self.compute_instrument_transitions(oldblock, newblock))

        if components:
            return TransitionBlock(components, start_time)
        else:
            return None

    def compute_instrument_transitions(self, oldblock, newblock):
        components = {}
        for conf_name, old_conf in oldblock.configuration.items():
            if conf_name in newblock.configuration:
                conf_times = self.instrument_reconfig_times.get(conf_name,
                                                                None)
                if conf_times is not None:
                    new_conf = newblock.configuration[conf_name]
                    ctime = conf_times.get((old_conf, new_conf), None)
                    def_time = conf_times.get('default', None)
                    if ctime is not None:
                        s = '{0}:{1} to {2}'.format(conf_name, old_conf,
                                                    new_conf)
                        components[s] = ctime
                    elif def_time is not None and not old_conf == new_conf:
                        s = '{0}:{1} to {2}'.format(conf_name, old_conf,
                                                    new_conf)
                        components[s] = def_time

        return components
